import os
from functools import partial

os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.85")
# No BFC preallocation: at 30B the fp32 masters pin the middle of the pool
# during sampling and fragment it below the trainer step's contiguous needs.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P, reshard
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import losses
import models
import prompting
import timing
import utils
from models import quant, sampling
from tasks import rollout


DEFAULT_CFG = """
model:
  source: Qwen/Qwen3-1.7B
  weights_dir: ~/.cache/postax/weights
  init: pretrained
  hidden_size: null
  num_hidden_layers: null
train:
  shard:
    dp: null
    tp: 1
    fsdp: true
dtypes:
  sampler: bfloat16
  trainer: bfloat16
  master: float32
sampler:
  shard:
    dp: null
    tp: null
  staleness: 0
  weight_noise_scale: 0.0  # fixed relative Gaussian weight perturbation (synthetic TIM)
  weight_quant: null  # weight-only sampler quant: null | int8 | fp8 | int4 | fp4 | intN (N in 2..16)
  weight_quant_group: 0  # absmax scale block along contraction axes (0 = per-channel)
  kv_quant_bits: 0  # fake quant of sampler KV-cache writes: 0 (off) | 2..16 (int-N) | fp8 (e4m3, unit scale) | fp6/fp4 (scaled mini-float)
  vocab_logprobs: 128  # null | positive integer (top-k) | full
rl:
  num_prompts: 16
  group_size: 8
  seq_len: 1024
  reward_mode: group_centered
  sc: false  # uses sampler.vocab_logprobs: exact if full, modeled tail if top-k
  is:  # generic IS correction on the trainer/sampler ratio p/q; the named
       # methods (TIS/CISPO, MIS, IcePop, PPO/DAPO, GSPO, TOPR) are points in
       # this grid — spelling table in README.md
    level: null    # null | token | seq_mean (geometric mean) | seq_sum (full product)
    stat: ratio    # ratio | tv: what the band tests — the ratio r = p/q, or the
                   # binary TV divergence |q - p| (DPPO; high = delta, e.g. 0.2)
    low: 0.0       # trust band on the ratio; 0 / inf leave a side open
    high: .inf     # .inf is YAML for float inf; CLI overrides accept plain inf
    outside: drop  # clamp = pull weight to the nearest bound | drop = zero it
    inside: ratio  # ratio = keep the IS weight | one = plain gradient
    sign: both     # both | neg | pos = which advantages get corrected (rest w=1)
                   # | ppo = PPO band: pos (0, high], neg [low, 3.0] (dual clip)
  minibatches: 1  # optimizer steps per rollout batch: contiguous group-aligned
                  # slices in fixed order (PPO-style multi-step)
env:
  id: countdown
  args: null
  # auto is only a naming-convention convenience. Experiment sweeps should
  # explicitly select text for Base weights and chat for post-trained weights.
  prompt_format: auto
  chat_template_kwargs:
    enable_thinking: false
opt:
  optimizer: adamw
  lr: 1e-6
  b2: 0.99
  lr_warmup: 0.0
stop:
  steps: 100
  collapse: 0.5  # auto-stop after 10 steps below this fraction of peak reward (0 disables)
eval:
  enabled: true
  num_prompts: 128  # multiple of rl.num_prompts; held out from train if the env has no eval split
  every_steps: 50
log:
  project: postax-rl
  wandb_mode: online
  dir: ./outputs
  run_name: null
  save_model: false
  every_steps: 1
run:
  seed: 0
"""


def resolve_shard(shard, default_tp, name):
    tensor_parallel = default_tp if shard.tp is None else shard.tp
    if jax.device_count() % tensor_parallel:
        raise ValueError(
            f"{name}.tp={tensor_parallel} does not divide "
            f"device_count={jax.device_count()}")
    data_parallel = jax.device_count() // tensor_parallel
    if shard.dp is not None and shard.dp != data_parallel:
        raise ValueError(
            f"{name}.dp={shard.dp}, but chips/tp={data_parallel}")
    return data_parallel, tensor_parallel


def fixed_weight_delta(weights, key, scale):
    """Fixed relative Gaussian perturbation, generated once per run."""
    leaves, structure = jax.tree.flatten(weights)
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(structure, [
        scale * jax.random.normal(
            k, w.shape, w.dtype, out_sharding=w.sharding) * w
        for k, w in zip(keys, leaves)
    ])


def train(cfg):
    key = jax.random.key(cfg.run.seed)
    sampler_dtype = jnp.dtype(cfg.dtypes.sampler)
    quantized = sampler_dtype.itemsize == 1
    sampler_float_dtype = jnp.bfloat16 if quantized else sampler_dtype
    _, trainer_tp = resolve_shard(cfg.train.shard, 1, "train.shard")
    with timing.context("load model"):
        model = models.load(
            cfg.model.source,
            cfg.model.weights_dir,
            tp_size=trainer_tp,
            dp_shard=cfg.train.shard.fsdp,
            init=cfg.model.init,
            hidden_size=cfg.model.hidden_size,
            num_hidden_layers=cfg.model.num_hidden_layers,
            key=jax.random.fold_in(key, 1),
        )
    chat_template_kwargs = OmegaConf.to_container(cfg.env.chat_template_kwargs)
    prompt_format = prompting.resolve_prompt_format(
        cfg.env.prompt_format, cfg.model.source)
    prompting.validate_prompt_contract(
        model.tokenizer, prompt_format, chat_template_kwargs)
    # Persist the resolved protocol in W&B/run metadata rather than the
    # convenience spelling "auto".
    cfg.env.prompt_format = prompt_format
    model.weights = jax.tree.map(
        lambda weight: weight.astype(jnp.dtype(cfg.dtypes.master)),
        model.weights,
    )
    trainer_forward = partial(model.forward, dtype=jnp.dtype(cfg.dtypes.trainer))
    trainer_mesh = jax.tree.leaves(model.weights)[0].sharding.mesh
    jax.set_mesh(trainer_mesh)

    sampler_dp, sampler_tp = resolve_shard(cfg.sampler.shard, jax.device_count(), "sampler.shard")
    sampler_mesh = jax.make_mesh(
        (sampler_dp, sampler_tp),
        ("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    sampler_shardings = jax.tree.map(
        lambda weight: NamedSharding(sampler_mesh, weight.sharding.spec),
        model.weights,
    )
    copy_sampler_weights = jax.jit(
        lambda weights: {
            key: weight.astype(sampler_float_dtype)
            if weight.ndim > 1 and "norm" not in key else weight
            for key, weight in weights.items()
        },
        out_shardings=sampler_shardings,
    )

    weight_noise = None
    if cfg.sampler.weight_noise_scale:
        weight_noise = copy_sampler_weights(fixed_weight_delta(
            model.weights,
            jax.random.fold_in(key, 2),
            cfg.sampler.weight_noise_scale,
        ))

    def prepare_sampler_weights(weights, delta):
        if delta is not None:
            weights = jax.tree.map(
                lambda weight, delta: weight + delta,
                weights,
                delta,
            )
        if cfg.sampler.weight_quant is not None:
            weights = quant.fake_quantize_tree(
                weights,
                quant.weight_quant_spec(cfg.sampler.weight_quant),
                cfg.sampler.weight_quant_group,
            )
        if quantized:
            weights = quant.quantize_tree(weights, sampler_dtype)
        return weights
    prepare_sampler_weights = jax.jit(
        prepare_sampler_weights,
        donate_argnames=("weights",) if not quantized else (),
    )

    def sampler_snapshot(weights):
        weights = copy_sampler_weights(jax.block_until_ready(weights))
        with jax.set_mesh(sampler_mesh):
            return prepare_sampler_weights(weights, weight_noise)

    sampler_forward = partial(
        model.forward, dtype=sampler_float_dtype,
        kv_quant_bits=cfg.sampler.kv_quant_bits)
    sampler_init_kv = partial(model.init_kv, dtype=sampler_float_dtype)
    # Eval always samples clean bf16 (no weight/KV quant, no noise): it
    # measures what the policy learned, not how the sampler executes it.
    eval_forward = partial(model.forward, dtype=jnp.bfloat16)

    total_steps = cfg.stop.steps
    # The LR schedule counts optimizer steps, not rollouts.
    optimizer, optimizer_state = utils.make_optimizer(
        cfg.opt, total_steps * cfg.rl.minibatches, model.weights)
    run_dir = utils.start_run(cfg)
    print(f"n_params={sum(w.size for w in jax.tree.leaves(model.weights)):_}")

    is_cfg = (losses.ISCorrection(**OmegaConf.to_container(cfg.rl["is"]))
              if cfg.rl["is"].level is not None else None)

    @partial(jax.jit, static_argnames="start",
             donate_argnames=("weights", "optimizer_state"))
    def optimize(weights, optimizer_state, batch, key, start=0):
        if cfg.rl.minibatches > 1:
            # Slice inside jit: the partitioner emits a local shard-aligned
            # slice; the same slice eagerly all-gathers the full-vocab array.
            rows = num_prompts * group_size // cfg.rl.minibatches
            batch = jax.tree.map(
                lambda x: x.at[start:start + rows].get(
                    out_sharding=jax.typeof(x).sharding),
                batch)

        def objective(weights):
            # Padding rows are dead weight in the MoE dispatch: mask them out.
            hidden = trainer_forward(
                batch["tokens"], weights, logits=False,
                live=batch["tokens"] != model.tokenizer.pad_token_id)
            return losses.loss_fn_rl(
                hidden,
                lambda x: model.head(x, weights, dtype=jnp.dtype(cfg.dtypes.trainer)),
                batch,
                is_cfg,
                score_center=cfg.rl.sc,
            )

        (loss, rl_metrics), gradients = jax.value_and_grad(
            objective, has_aux=True)(weights)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, weights)
        weights = utils.apply_updates(weights, updates, key)
        return weights, optimizer_state, {
            "loss": loss,
            "grad_norm": optax.global_norm(gradients),
            "update_norm": optax.global_norm(updates),
            **rl_metrics,
        }

    env_args = OmegaConf.to_container(cfg.env.args) if cfg.env.args else {}
    environment = rollout.load(cfg.env.id, **env_args)
    if getattr(environment, "max_turns", 1) != 1:
        raise ValueError("synchronous RL supports single-turn environments")
    if environment.provides_advantages:
        raise ValueError(
            "synchronous RL expects environments to return scalar rewards")
    # Hub datasets are ordered by source; the sequential scan below would
    # otherwise turn source clusters into a difficulty curriculum.
    dataset = environment.get_dataset().shuffle(seed=cfg.run.seed)
    seq_len = cfg.rl.seq_len
    if not len(dataset):
        raise ValueError("environment training dataset is empty")
    preview_tokens = prompting.tokenize_prompt(
        dataset[0]["prompt"], model.tokenizer, prompt_format,
        chat_template_kwargs)[:seq_len]
    if jax.process_index() == 0:
        print(prompting.prompt_diagnostic(
            model.tokenizer, prompt_format, preview_tokens))
    num_prompts = cfg.rl.num_prompts
    group_size = cfg.rl.group_size
    token_sharding = NamedSharding(trainer_mesh, P("data", None))
    rollout_sharding = NamedSharding(trainer_mesh, P("data"))
    vocab_sharding = NamedSharding(trainer_mesh, P("data", None, "model"))
    center_sharding = NamedSharding(trainer_mesh, P("data", None, None))

    def sample(rows, weights, key, vocab_logprobs_spec, forward=None):
        inputs = [
            {**row, "example_id": example_id}
            for example_id, row in enumerate(rows)
            for _ in range(group_size)
        ]
        token_buffer, prompt_lengths = rollout.prompt_buffer(
            inputs, model.tokenizer, seq_len,
            prompt_format=prompt_format,
            chat_template_kwargs=chat_template_kwargs)
        centering = {}
        with jax.set_mesh(sampler_mesh):
            (tokens, sampling_token_logprobs, sampling_vocab_logprobs,
             head_logprobs, head_ids) = sampling.generate(
                key, model, token_buffer, prompt_lengths, weights=weights,
                forward=forward or sampler_forward, init_kv=sampler_init_kv,
                logprobs_dtype=(jnp.float32
                                if vocab_logprobs_spec == "full" else None),
                logprobs_topk=(vocab_logprobs_spec
                               if isinstance(vocab_logprobs_spec, int) else 0))
            if head_ids is not None:
                # The transported head feeds both kl_head diagnostics and SC.
                centering = {"center_ids": head_ids,
                             "center_logprobs": head_logprobs}
            host_tokens = np.asarray(tokens)
        tokens = reshard(tokens, token_sharding)
        sampling_token_logprobs = reshard(
            sampling_token_logprobs, token_sharding)
        mask, completion_lengths, texts = rollout.completions(host_tokens, prompt_lengths, model.tokenizer)
        rewards = rollout.score(environment.rubric, inputs, texts, group_size)
        # Answer-extraction rate; separates policy format shifts from scorer
        # failures (c10 record). environment.parser, not rubric.parser.
        extract = getattr(environment.parser, "extract_fn", str)
        frac_parsed = np.mean([
            bool(answer := extract(text)) and answer != text for text in texts])
        truncated = (
            (prompt_lengths + completion_lengths >= seq_len)
            & (host_tokens[:, -1] != model.tokenizer.eos_token_id)
        )
        return (
            tokens, sampling_token_logprobs, sampling_vocab_logprobs,
            mask, rewards, completion_lengths, truncated, centering, frac_parsed)

    eval_rows = []
    if cfg.eval.enabled:
        # Hold out the head of the (shuffled) train set — deliberately NOT
        # the env's own eval split, which can be a different distribution
        # (intellect-math 0.1.7 ships 81K unbanded rows; c10 record).
        eval_dataset = dataset.select(range(cfg.eval.num_prompts))
        dataset = dataset.select(range(cfg.eval.num_prompts, len(dataset)))
        eval_rows = [
            eval_dataset[index]
            for index in range(len(eval_dataset))
        ]
        if len(eval_rows) % num_prompts:
            raise ValueError(
                "eval.num_prompts must be a multiple of rl.num_prompts "
                "(eval runs in train-shaped chunks)")
    eval_key = jax.random.fold_in(key, 3)

    data_parallel = trainer_mesh.shape["data"]
    if num_prompts // cfg.rl.minibatches * group_size % data_parallel:
        raise ValueError(
            "rl.num_prompts / rl.minibatches * rl.group_size (one optimizer "
            f"slice) must be divisible by data parallelism ({data_parallel})")
    if eval_rows and len(eval_rows) * group_size % data_parallel:
        raise ValueError(
            "eval.num_prompts * rl.group_size must be divisible by "
            f"data parallelism ({data_parallel})")
    refresh_period = cfg.sampler.staleness + 1
    peak_reward = collapse_steps = 0

    for step in tqdm(range(total_steps), desc="RL", disable=jax.process_index() != 0):
        if step % refresh_period == 0:
            sampler_weights = sampler_snapshot(model.weights)
        should_log = step % cfg.log.every_steps == 0 or step == total_steps - 1
        eval_metrics = {}
        if should_log and eval_rows and (
                step % cfg.eval.every_steps == 0
                or step == total_steps - 1):
            eval_weights = copy_sampler_weights(jax.block_until_ready(model.weights))
            # Eval never feeds a trainer step, so no distribution payload is
            # consumed regardless of the training sampler's transport spec.
            chunks = [
                sample(eval_rows[start:start + num_prompts], eval_weights,
                       jax.random.fold_in(eval_key, start),
                       vocab_logprobs_spec=None,
                       forward=eval_forward)
                for start in range(0, len(eval_rows), num_prompts)
            ]
            del eval_weights
            eval_metrics = {
                "eval_reward": np.mean([c[4].mean() for c in chunks]),
                "eval_mean_completion_len": np.mean([c[5].mean() for c in chunks]),
                "eval_frac_parsed": np.mean([c[8] for c in chunks]),
            }
        rows = [
            dataset[(step * num_prompts + offset) % len(dataset)]
            for offset in range(num_prompts)
        ]
        key, sample_key = jax.random.split(key)
        (tokens, sampling_token_logprobs, sampling_vocab_logprobs, mask, rewards,
         completion_lengths, truncated, centering, frac_parsed) = sample(
             rows, sampler_weights, sample_key, cfg.sampler.vocab_logprobs)
        advantages = losses.compute_advantages(rewards, group_size, cfg.rl.reward_mode)
        # stop run if reward collapses
        reward = rewards.mean()
        peak_reward = max(peak_reward, reward)
        collapse_steps = collapse_steps + 1 if reward < cfg.stop.collapse * peak_reward else 0
        collapsed = collapse_steps == 10
        batch = {
            "tokens": tokens,
            "mask": jax.device_put(mask, token_sharding),
            "advantages": jax.device_put(advantages, rollout_sharding),
            "sampling_token_logprobs": sampling_token_logprobs,
            **{name: reshard(value, center_sharding)
               for name, value in centering.items()},
        }
        if refresh_period == 1:
            del sampler_weights
        if sampling_vocab_logprobs is not None:
            # Reshard to the trainer mesh only once the sampler weights can be
            # dropped: source + destination + collective scratch is ~3x the
            # 9.3 GiB/device array and sets the peak between steps.
            batch["sampling_vocab_logprobs"] = reshard(
                sampling_vocab_logprobs, vocab_sharding)
            del sampling_vocab_logprobs
        # One optimizer step per contiguous group-aligned slice, fixed order.
        # Advantages are already computed over the full batch: slices are
        # whole groups, so group centering is unaffected.
        slice_rows = num_prompts * group_size // cfg.rl.minibatches
        inner_metrics = []
        for start in range(0, num_prompts * group_size, slice_rows):
            key, optimizer_key = jax.random.split(key)
            model.weights, optimizer_state, m = optimize(
                model.weights, optimizer_state, batch, optimizer_key,
                start=start)
            inner_metrics.append(m)
        del batch

        if should_log:
            optimizer_metrics = {
                name: np.mean([np.asarray(m[name]) for m in inner_metrics])
                for name in inner_metrics[0]}
            metrics = {
                "step": step,
                "reward": reward,
                "mean_completion_len": completion_lengths.mean(),
                "frac_nonzero_advantage": (advantages != 0).mean(),
                "frac_parsed": frac_parsed,
                "frac_truncated": truncated.mean(),
                "sampler_age": step % refresh_period,  # rollouts, not opt steps
                "collapsed": collapsed,
                "weight_norm": utils.weight_norm(model.weights),
                **optimizer_metrics,
                **eval_metrics,
            }
            utils.log_metrics(step, metrics, run_dir)
        if collapsed:
            print(f"Stopping after {collapse_steps} steps below "
                  f"{cfg.stop.collapse}x peak reward")
            break

    if cfg.log.save_model:
        utils.save_model(model, run_dir, "final_model")
    if jax.process_index() == 0:
        wandb.finish()


def main():
    # Spawn, never fork: forked scorer workers inherit this threaded CUDA
    # process and silently corrupt (c10 record).
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    cfg = utils.parse_cfg(DEFAULT_CFG)
    if cfg.sampler.staleness < 0:
        raise ValueError("sampler.staleness must be non-negative")
    if cfg.sampler.weight_noise_scale < 0:
        raise ValueError("sampler.weight_noise_scale must be non-negative")
    if cfg.sampler.kv_quant_bits not in (0, "fp8", "fp6", "fp4", *range(2, 17)):
        raise ValueError(
            "sampler.kv_quant_bits must be 0, in [2, 16], or 'fp8'/'fp6'/'fp4'")
    vocab_logprobs = cfg.sampler.vocab_logprobs
    if not (
            vocab_logprobs is None
            or vocab_logprobs == "full"
            or (isinstance(vocab_logprobs, int)
                and not isinstance(vocab_logprobs, bool)
                and vocab_logprobs > 0)):
        raise ValueError(
            "sampler.vocab_logprobs must be null, a positive integer, or 'full'")
    if not isinstance(cfg.rl.sc, bool):
        raise ValueError("rl.sc must be true or false")
    if cfg.rl.sc and vocab_logprobs is None:
        raise ValueError("rl.sc=true requires sampler.vocab_logprobs")
    if cfg.sampler.weight_quant is not None:
        quant.weight_quant_spec(cfg.sampler.weight_quant)  # raises on unknown schemes
    if cfg.sampler.weight_quant_group < 0:
        raise ValueError("sampler.weight_quant_group must be non-negative")
    isc = cfg.rl["is"]
    if isinstance(isc.high, str):
        # CLI yaml reads bare 'inf' as a string ('.inf' is the yaml float).
        isc.high = float(isc.high)
    if isc.level is not None:
        if isc.level not in losses.IS_LEVELS:
            raise ValueError(f"rl.is.level must be null or one of {losses.IS_LEVELS}")
        if isc.outside not in losses.IS_OUTSIDE:
            raise ValueError(f"rl.is.outside must be one of {losses.IS_OUTSIDE}")
        if isc.inside not in losses.IS_INSIDE:
            raise ValueError(f"rl.is.inside must be one of {losses.IS_INSIDE}")
        if isc.stat not in losses.IS_STATS:
            raise ValueError(f"rl.is.stat must be one of {losses.IS_STATS}")
        if isc.sign not in losses.IS_SIGNS:
            raise ValueError(f"rl.is.sign must be one of {losses.IS_SIGNS}")
        if not 0 <= isc.low <= isc.high:
            raise ValueError("rl.is bounds must satisfy 0 <= low <= high")
        if isc.stat == "tv":
            if isc.level == "seq_sum":
                raise ValueError("rl.is.level=seq_sum requires rl.is.stat=ratio")
            if isc.outside != "drop" or isc.low != 0:
                raise ValueError(
                    "rl.is.stat=tv gates on the divergence: outside=drop, low=0")
            if not 0 < isc.high <= 1:
                raise ValueError(
                    "rl.is.high is the TV threshold delta, in (0, 1]")
        elif isc.outside == "drop" and not isc.low <= 1 <= isc.high:
            raise ValueError(
                "rl.is drop band must contain the on-policy ratio 1")
        if isc.outside == "clamp" and isc.inside == "one":
            raise ValueError(
                "rl.is outside=clamp requires inside=ratio (a clamped "
                "out-of-band weight would exceed the in-band weight 1)")
        if isc.sign == "ppo" and (isc.outside != "drop" or isc.inside != "ratio"):
            raise ValueError(
                "rl.is.sign=ppo is the PPO surrogate: outside=drop, inside=ratio")
    if cfg.rl.minibatches < 1 or cfg.rl.num_prompts % cfg.rl.minibatches:
        raise ValueError("rl.minibatches must be >= 1 and divide rl.num_prompts")
    if cfg.rl.sc:
        if isc.level is not None and isc.stat != "ratio":
            raise ValueError(
                "score centering composes only with rl.is.stat=ratio")
    if cfg.rl.group_size < 1 or cfg.rl.num_prompts < 1:
        raise ValueError("rl.group_size and rl.num_prompts must be positive")
    if cfg.stop.steps < 1:
        raise ValueError("stop.steps must be positive")
    if not 0 <= cfg.stop.collapse <= 1:
        raise ValueError("stop.collapse must be in [0, 1] (0 disables)")
    if cfg.rl.reward_mode not in losses.RL_REWARD_MODES:
        raise ValueError(
            f"rl.reward_mode must be one of {losses.RL_REWARD_MODES}")
    train(cfg)


if __name__ == "__main__":
    main()
