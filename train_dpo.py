from functools import partial
from itertools import chain, count

from tasks import data
import jax
import jax.numpy as jnp
import optax
import timing
import wandb
from tqdm.auto import tqdm

from tasks.evals import run_evals
import losses
import models
import utils


DEFAULT_CFG = """
model:
  source: null
  weights_dir: ~/.cache/stx/weights
  init: pretrained
  tp_size: 1
  dp_shard: false
  hidden_size: null
  num_hidden_layers: null
train:
  dataset: null
  split: null
  seq_len: null
  batch_size: null
dpo:
  beta: 0.1
  ref_weight: 1.0
  margin: 0.0
  sft_weight: 0.0
  norm: false
  stop_grad_rejected: false
reg:
  dataset: null
  split: null
  seq_len: null
  batch_size: null
  method: null
  coeff: 0.0
  synth: false
  buffer: 1
pref_valid:
  dataset: null
  split: null
  seq_len: null
  batch_size: null
opt:
  n_epochs: 1
  tokens: null
  optimizer: adam
  lr: 1e-6
  b2: 0.997
  lr_warmup: 0.0
log:
  project: stx
  wandb_mode: online
  dir: ./outputs
  run_name: null
  evals: []
  loss_steps: 100
  eval_steps: 20
  num_tokens_eval: 1000000
run:
  seed: 0
"""


def train(cfg) -> None:
    _, key_model = jax.random.split(jax.random.key(cfg.run.seed))
    with timing.context("load model"):
        model = models.load(
            str(cfg.model.source),
            cfg.model.weights_dir,
            tp_size=cfg.model.tp_size,
            dp_shard=cfg.model.dp_shard,
            init=cfg.model.init,
            hidden_size=cfg.model.hidden_size,
            num_hidden_layers=cfg.model.num_hidden_layers,
            key=key_model,
        )
    model.weights = jax.tree.map(lambda x: x.astype(jnp.float32), model.weights)
    model.forward = jax.jit(model.forward)
    n_params = sum(w.size for w in jax.tree.leaves(model.weights))
    print(f"{n_params=:_}")
    weights_pi = jax.tree.map(lambda x: x.astype(jnp.bfloat16), model.weights)
    with timing.context("load datasets"):
        train_len, make_train, _ = data.load(
            cfg.train.dataset, cfg.train.split, cfg.train.batch_size, cfg.train.seq_len, model.tokenizer, seed=cfg.run.seed
        )
        _, make_reg, _ = (
            (None, None, None)
            if cfg.reg.dataset is None
            else data.load(
                cfg.reg.dataset, cfg.reg.split, cfg.reg.batch_size, cfg.reg.seq_len, model.tokenizer, seed=cfg.run.seed
            )
        )
        _, make_pref_valid, _ = (
            (None, None, None)
            if cfg.pref_valid.dataset is None
            else data.load(
                cfg.pref_valid.dataset, cfg.pref_valid.split, cfg.pref_valid.batch_size, cfg.pref_valid.seq_len, model.tokenizer, seed=cfg.run.seed
            )
        )
    ds_train_eval = chain.from_iterable(make_train(epoch) for epoch in count())
    ds_reg = None if make_reg is None else chain.from_iterable(make_reg(epoch) for epoch in count())
    ds_reg_eval = None if make_reg is None else chain.from_iterable(make_reg(epoch) for epoch in count())
    tokens_per_batch, total_steps, target_tokens, streaming_train = utils.compute_training_schedule(train_len, cfg.train, cfg.opt)
    optimizer, opt_state = utils.make_optimizer(cfg.opt, total_steps, model.weights)
    run_dir = utils.start_run(cfg)
    synth_key = jax.random.key(cfg.run.seed)
    synth_reg_iter = iter(())

    def loss_fn(weights_theta, weights_pi, batch_train, batch_reg=None):
        primary_loss_avg, primary_loss_sum, primary_num_tokens = losses.loss_fn_dpo(model.forward, weights_theta, weights_pi, batch_train, cfg.dpo)
        reg_loss_avg = losses.loss_fn_reg(model.forward, weights_theta, weights_pi, batch_train, batch_reg, cfg.reg, primary_loss_avg.dtype)
        optim_loss_avg = primary_loss_avg + cfg.reg.coeff * reg_loss_avg
        return optim_loss_avg, (reg_loss_avg, primary_loss_sum, primary_num_tokens)

    @partial(jax.jit, donate_argnames=("weights_theta", "opt_state"))
    def opt_step(weights_theta, weights_pi, opt_state, batch_train, batch_reg=None):
        (optim_loss_avg, (reg_loss_avg, primary_loss_sum, primary_num_tokens)), grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(weights_theta, weights_pi, batch_train, batch_reg)
        updates, opt_state = optimizer.update(grads, opt_state, weights_theta)
        weights_theta = optax.apply_updates(weights_theta, updates)
        batch_losses = {"optim_loss_avg": optim_loss_avg, "reg_loss_avg": reg_loss_avg, "primary_loss_sum": primary_loss_sum, "primary_num_tokens": primary_num_tokens}
        return weights_theta, opt_state, batch_losses

    pending_metrics = None
    step = masked_tokens_seen = total_tokens_seen = 0
    loss_every_tokens = target_tokens / cfg.log.loss_steps
    eval_every_tokens = target_tokens / cfg.log.eval_steps
    next_loss_at = next_eval_at = 0
    interval_reg_loss_avg_sum = interval_optim_loss_avg_sum = interval_primary_loss_sum = 0.0
    interval_steps = interval_primary_tokens = epoch_tokens = code_len_model_and_data = 0
    train_batches = ((epoch, batch_train) for epoch in (count() if streaming_train else range(cfg.opt.n_epochs)) for batch_train in make_train(epoch))
    pbar = tqdm(total=total_steps, desc="Train") if jax.process_index() == 0 else None

    for epoch, batch_train in train_batches:

        # logging
        if total_tokens_seen >= next_loss_at:
            while total_tokens_seen >= next_loss_at:
                next_loss_at += loss_every_tokens
            eval_metrics = {
                "step": step,
                "epoch": epoch,
                "masked_tokens_seen": masked_tokens_seen,
                "total_tokens_seen": total_tokens_seen,
                "weight_norm": utils.weight_norm(model.weights),
                "epiplexity": 0,
            }
            if step > 0:
                interval_reg_loss_avg = interval_reg_loss_avg_sum / interval_steps
                primary_loss_avg = interval_primary_loss_sum / interval_primary_tokens
                code_len_data_given_model = primary_loss_avg * min(masked_tokens_seen, epoch_tokens)
                code_len_model = code_len_model_and_data - code_len_data_given_model
                eval_metrics |= {
                    "loss/optim": interval_optim_loss_avg_sum / interval_steps,
                    "loss/primary": primary_loss_avg,
                    "loss/regularization": interval_reg_loss_avg,
                    "loss/regularization_weighted": cfg.reg.coeff * interval_reg_loss_avg,
                    "epiplexity": code_len_model,
                }
                interval_reg_loss_avg_sum = interval_optim_loss_avg_sum = interval_primary_loss_sum = 0.0
                interval_steps = interval_primary_tokens = 0
            if total_tokens_seen >= next_eval_at:
                while total_tokens_seen >= next_eval_at:
                    next_eval_at += eval_every_tokens
                eval_metrics |= run_evals(cfg.log.evals, model, weights_pi, ds_train_eval, ds_reg_eval, make_pref_valid, run_dir, step, epoch, tokens_per_batch, cfg.train.split, cfg.log.num_tokens_eval)
            if pending_metrics is not None: utils.log_metrics(*pending_metrics, run_dir)
            pending_metrics = step, eval_metrics

        # train step
        batch_reg = None
        if cfg.reg.method in ("ntp_pre", "kl_pre"):
            if cfg.reg.synth:
                try:
                    batch_reg = next(synth_reg_iter)
                except StopIteration:
                    synth_key, key_reg = jax.random.split(synth_key)
                    synth_reg_batches = utils.sample_synthetic_lm_batches(
                        key_reg, model, weights_pi, cfg.reg.batch_size, cfg.reg.seq_len, cfg.reg.buffer
                    )
                    synth_reg_iter = (jax.tree.map(lambda x: x[i], synth_reg_batches) for i in range(cfg.reg.buffer))
                    batch_reg = next(synth_reg_iter)
            else:
                batch_reg = next(ds_reg)
        model.weights, opt_state, batch_losses = opt_step(model.weights, weights_pi, opt_state, batch_train, batch_reg)

        # logging
        interval_optim_loss_avg_sum += batch_losses["optim_loss_avg"]
        interval_reg_loss_avg_sum += batch_losses["reg_loss_avg"]
        interval_primary_loss_sum += batch_losses["primary_loss_sum"]
        interval_primary_tokens += batch_losses["primary_num_tokens"]
        if epoch == 0:
            epoch_tokens += batch_losses["primary_num_tokens"]
            code_len_model_and_data += batch_losses["primary_loss_sum"]
        step += 1
        interval_steps += 1
        masked_tokens_seen += batch_losses["primary_num_tokens"]
        total_tokens_seen += tokens_per_batch
        if pbar is not None:
            pbar.update(1)

        if total_tokens_seen >= target_tokens:
            break

    if pbar is not None:
        pbar.close()
    utils.log_metrics(*pending_metrics, run_dir)
    if jax.process_index() == 0: wandb.finish()


def main() -> None:
    train(utils.parse_cfg(DEFAULT_CFG))


if __name__ == "__main__":
    main()
