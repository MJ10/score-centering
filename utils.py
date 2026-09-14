import json
import operator as op
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import optax
import wandb
from jax.sharding import PartitionSpec as P

from models import sampling


def parse_cfg(default_config):
    cfg = OmegaConf.create(default_config)
    OmegaConf.set_struct(cfg, True)
    if "env" in cfg and cfg.env.args is not None:
        # Free-form load_environment kwargs; the keyset varies by environment.
        OmegaConf.set_struct(cfg.env.args, False)
    overrides = OmegaConf.from_cli()
    primary = jax.process_index() == 0
    if primary:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            project=OmegaConf.select(overrides, "log.project", default=cfg.log.project),
            mode=OmegaConf.select(overrides, "log.wandb_mode", default=cfg.log.wandb_mode),
            name=OmegaConf.select(overrides, "log.run_name", default=cfg.log.run_name),
        )
        print(f"{wandb.run.id=}")
    cfg = OmegaConf.merge(cfg, overrides)
    if primary:
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
    return cfg


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def sample_synthetic_lm_batches(key, model, weights_pi, batch_size, seq_len, num_batches):
    if model.config.get("model_type") != "llama":
        raise ValueError("reg.synth currently supports Llama models only")
    if batch_size is None or seq_len is None:
        raise ValueError("reg.synth requires reg.batch_size and reg.seq_len")
    bos_id = model.tokenizer.bos_token_id
    if bos_id is None:
        raise ValueError("reg.synth requires tokenizer.bos_token_id")

    tokens = jnp.full((num_batches * batch_size, seq_len), model.tokenizer.pad_token_id,
                      dtype=jnp.int32)
    tokens = tokens.at[:, 0].set(bos_id)
    tokens, _, _, _, _ = sampling.generate(
        key, model, tokens, jnp.ones(tokens.shape[0], jnp.int32),
        weights=weights_pi, eos_id=-1)
    tokens = jnp.reshape(tokens, (num_batches, batch_size, seq_len),
                         out_sharding=P(None, "data", None))
    return {"tokens": tokens, "mask": jnp.ones_like(tokens, dtype=bool)}


def compute_training_schedule(train_len, train, opt):
    tokens_per_batch = train.batch_size * train.seq_len
    step_limit = None if opt.tokens is None else max(1, (opt.tokens + tokens_per_batch - 1) // tokens_per_batch)
    streaming_train = train_len is None
    if streaming_train:
        if opt.tokens is None:
            raise ValueError("Streaming train datasets require tokens to be set")
        total_steps = step_limit
    else:
        max_batches = train_len // train.batch_size
        total_steps = opt.n_epochs * max_batches if step_limit is None else min(opt.n_epochs * max_batches, step_limit)
    target_tokens = total_steps * tokens_per_batch if opt.tokens is None else min(opt.tokens, total_steps * tokens_per_batch)
    return tokens_per_batch, total_steps, target_tokens, streaming_train


def make_optimizer(opt, total_steps, weights, tokens_per_batch=None):
    warmup_steps = (
        (opt.warmup_tokens + tokens_per_batch - 1) // tokens_per_batch
        if "warmup_tokens" in opt else int(opt.lr_warmup * total_steps)
    )
    lr_schedule = optax.join_schedules(
        [
            optax.linear_schedule(0, opt.lr, warmup_steps),
            optax.constant_schedule(opt.lr),
        ],
        [warmup_steps],
    )
    if opt.optimizer == "sgd":
        optimizer = optax.sgd(lr_schedule)
    elif opt.optimizer == "adamw":
        optimizer = optax.adamw(lr_schedule, 0.9, opt.b2, weight_decay=0.02)
    elif opt.optimizer == "adafactor":
        optimizer = optax.adafactor(lr_schedule, decay_rate=opt.b2)
    else:
        raise ValueError(f"Unknown optimizer: {opt.optimizer}")
    return optimizer, optimizer.init(weights)


def start_run(train_config):
    train_config = OmegaConf.to_container(train_config, resolve=True)
    primary = jax.process_index() == 0
    resolved_run_name = train_config["log"]["run_name"] or (wandb.run.id if primary else "run")
    run_dir = Path(train_config["log"]["dir"]).expanduser() / train_config["log"]["project"] / resolved_run_name
    if primary:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(run_dir / "config.json", train_config)
        print(f"run_dir={run_dir}")
    return run_dir


def save_model(model, run_dir, name="final_model"):
    if jax.process_index() == 0:
        output_dir = Path(run_dir) / name
        model.save(model, output_dir)
        print(f"saved_model_dir={output_dir}")


def weight_norm(weights):
    # f32 accumulation: a bf16 result is too coarse to show the norm drifting.
    return jax.tree.reduce_associative(
        op.add, jax.tree.map(lambda w: (w.astype(jnp.float32) ** 2).sum(), weights)) ** 0.5


def _stochastic_round_bf16(x, key):
    """f32 -> bf16 rounded up/down with probability given by the remainder.

    Stays entirely in 32-bit types: 16-bit random bits / uint16 bitcasts need
    width-changing reshapes that the TPU compiler lowers through its
    convolution emitter, whose fusion cost model overflows the stack on
    fused graphs the size of an opt_step. Zeroing the low mantissa bits makes
    the final f32 -> bf16 convert exact.
    """
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    rnd = jax.random.bits(key, x.shape, jnp.uint32, out_sharding=jax.typeof(x).sharding.spec)
    bits = (bits + (rnd & jnp.uint32(0xFFFF))) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)


def apply_updates(params, updates, key):
    """optax.apply_updates, with stochastic rounding for bfloat16 params.

    Round-to-nearest freezes bf16 masters whenever the per-element step is
    below half an ULP (~2e-3 relative) -- RL-size learning rates always are,
    and without momentum the rounded-away updates never accumulate.
    Stochastic rounding keeps the expected update instead.
    """
    leaves, treedef = jax.tree.flatten(params)
    if not any(l.dtype == jnp.bfloat16 for l in leaves):
        return optax.apply_updates(params, updates)
    keys = jax.tree.unflatten(treedef, list(jax.random.split(key, len(leaves))))

    def apply(p, u, k):
        if p.dtype != jnp.bfloat16:
            return (p + u).astype(p.dtype)
        return _stochastic_round_bf16(p.astype(jnp.float32) + u.astype(jnp.float32), k)

    return jax.tree.map(apply, params, updates, keys)


def round_mantissa(x, drop):
    b = jax.lax.bitcast_convert_type(x.astype(jnp.bfloat16), jnp.uint16)
    b = (b + jnp.uint16(1 << (drop - 1))) & jnp.uint16(0x10000 - (1 << drop))
    return jax.lax.bitcast_convert_type(b, jnp.bfloat16).astype(x.dtype)


def no_grad(forward):
    return lambda *args: forward(*jax.tree.map(jax.lax.stop_gradient, args))


def log_metrics(step, metrics, run_dir=None):
    metrics = jax.tree.map(lambda x: x.item() if hasattr(x, "item") else x, metrics)
    if jax.process_index() == 0:
        print(f"Step {step}", metrics)
        wandb.log(metrics, step=step, commit=True)
        if run_dir is not None:
            with open(Path(run_dir) / "logs.jsonl", "a") as f:
                f.write(json.dumps({"step": step, **metrics}, default=str) + "\n")
