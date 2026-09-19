# Score Centering

Official repository for the paper *[Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807)*

[![](https://img.shields.io/badge/arXiv-2609.20807-b31b1b.svg)](https://arxiv.org/abs/2609.20807)

Score centering stabilizes RL under training-inference mismatch (TIM) by canceling *drift*. Score Centering can be used alone or applied on top of importance sampling.

![Score centering under progressively stronger sampler quantization.](assets/30b_math.png)

## Score Centering: loss function

We store the sampler's top-k logprobs (k=128) and model the tail with the trainer's distribution, rescaled to match the sampler's tail mass. Both k=128 and k=32 matched full score centering in our experiments.

The implementations below return a loss per token. `train_logp` has shape `[..., V]`; `samp_logp` and `topk_ids` have shape `[..., k]`. Sampled token IDs and logprobs have shape `[...]`; advantages broadcast to that shape. Use float32 logprobs, without renormalizing the top-k head, and retain the sampled token's logprob even when it falls outside the head.

Omit `weight_fn` for score centering alone, or pass `tis_weight` / `mis_weight` to compose it with truncated / masked importance sampling.

<details>
<summary><b>JAX</b></summary>

```python
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient as sg


def score_centering_loss(train_logp, samp_logp, topk_ids, sampled_token,
                         samp_token_logp, advantage,
                         weight_fn=jnp.ones_like, eps=1e-6):
    head_logp = jnp.take_along_axis(train_logp, topk_ids, axis=-1)
    token_logp = jnp.take_along_axis(
        train_logp, sampled_token[..., None], axis=-1)[..., 0]
    p_head, q_head = jnp.exp(head_logp), jnp.exp(samp_logp)
    p_tail = jnp.maximum(1 - p_head.sum(-1), eps)
    q_tail = jnp.maximum(1 - q_head.sum(-1), eps)
    rho = q_tail / p_tail
    alpha = rho * weight_fn(1 / rho)
    head_weight = weight_fn(jnp.exp(head_logp - samp_logp))
    residual = q_head * head_weight - alpha[..., None] * p_head
    correction = (sg(residual) * head_logp).sum(-1)
    token_weight = weight_fn(jnp.exp(token_logp - samp_token_logp))
    return -sg(advantage) * (sg(token_weight) * token_logp - correction)


tis_weight = lambda r: jnp.minimum(r, 2.0)
mis_weight = lambda r: jnp.where((r >= 0.5) & (r <= 5.0), r, 0.0)

# logits: [B, T, V]; advantages: [B]; mask: [B, T] (generated tokens only).
token_loss = score_centering_loss(
    jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1),
    samp_logp, topk_ids, sampled_token, samp_token_logp,
    advantages[:, None], weight_fn=tis_weight)
loss = (token_loss * mask).sum() / jnp.maximum(mask.sum(), 1)
```

</details>

<details>
<summary><b>PyTorch</b></summary>

```python
import torch


def score_centering_loss(train_logp, samp_logp, topk_ids, sampled_token,
                         samp_token_logp, advantage,
                         weight_fn=torch.ones_like, eps=1e-6):
    head_logp = train_logp.gather(-1, topk_ids)
    token_logp = train_logp.gather(-1, sampled_token[..., None])[..., 0]
    p_head, q_head = head_logp.exp(), samp_logp.exp()
    p_tail = (1 - p_head.sum(-1)).clamp_min(eps)
    q_tail = (1 - q_head.sum(-1)).clamp_min(eps)
    rho = q_tail / p_tail
    alpha = rho * weight_fn(1 / rho)
    head_weight = weight_fn((head_logp - samp_logp).exp())
    residual = q_head * head_weight - alpha[..., None] * p_head
    correction = (residual.detach() * head_logp).sum(-1)
    token_weight = weight_fn((token_logp - samp_token_logp).exp())
    return -advantage.detach() * (token_weight.detach() * token_logp - correction)


tis_weight = lambda r: r.clamp_max(2.0)
mis_weight = lambda r: torch.where((r >= 0.5) & (r <= 5.0), r, 0.0)

# logits: [B, T, V]; advantages: [B]; mask: [B, T] (generated tokens only).
# Token IDs must be torch.long; all tensors must be on the same device.
token_loss = score_centering_loss(
    logits.float().log_softmax(-1), samp_logp, topk_ids, sampled_token,
    samp_token_logp, advantages[:, None], weight_fn=tis_weight)
loss = (token_loss * mask).sum() / mask.sum().clamp_min(1)
```

</details>

Full training implementation: [`losses.py`](losses.py).

## Training framework: postax

Score centering was developed in **postax**, a JAX framework I built for LLM full-stack research. This repo includes the full framework, with features beyond score centering:

- pretraining, SFT, DPO, RL
- Llama, Qwen2, Qwen3, Qwen3-MoE (load + save HF checkpoint)
- TP, DP, FSDP, EP

Sampling is JIT-compiled, and sampler logprobs stay in HBM (no CPU roundtrip)!

## Example: RL run

Defaults live in each trainer; override them with `key=value` arguments. For Qwen3-0.6B on Countdown with score centering:

```bash
python train_rl.py model.source=Qwen/Qwen3-0.6B \
  env.id=countdown env.prompt_format=chat \
  rl.sc=true sampler.vocab_logprobs=128 \
  opt.optimizer=sgd opt.lr=0.01
```

Use `log.wandb_mode=disabled` to run without W&B. Models are cached in `~/.cache/postax/weights`; configurations and metrics go to `outputs/`. Pretraining/SFT and DPO use `train_lm.py` and `train_dpo.py`.

## Paper sweeps

Paper configurations are in [`sweeps/`](sweeps/). Each YAML is a W&B grid sweep for one figure:

| Sweep | Experiment | Runs per seed |
| --- | --- | ---: |
| [2b_online_vs_offline](sweeps/2b_online_vs_offline.yaml) | 1.7B Countdown: reward modes, online vs. offline | 6 |
| [06b_weight_noise](sweeps/06b_weight_noise.yaml) | 0.6B Countdown: synthetic weight noise | 36 |
| [06b_quant_stale](sweeps/06b_quant_stale.yaml) | 0.6B Countdown: quantization and staleness | 26 |
| [30b_math](sweeps/30b_math.yaml) | 30B INTELLECT-2 math: quantization | 30 |
| [topk](sweeps/topk.yaml) | Score centering with k=32, k=128, or full logprobs | 27 |

## Install

The project is a [uv](https://docs.astral.sh/uv/) project pinned to Python 3.13. From the repository root:

```bash
uv sync
```

On Linux this installs `jax[cuda12]`; elsewhere it installs CPU JAX. The math experiments additionally need the INTELLECT-2 environment:

```bash
uv sync --extra math
```

Run any command in the environment with `uv run`, e.g. `uv run python train_rl.py ...`.

## Running on Slurm clusters with cluv

The repo is set up for [cluv](https://github.com/mila-iqia/cluv), which syncs a uv project to one or more Slurm clusters over SSH and submits jobs there. Cluster settings (accounts, partitions, GPUs, environment variables) live in the `[tool.cluv]` section of [`pyproject.toml`](pyproject.toml); the job script is [`scripts/job.sh`](scripts/job.sh).

```bash
uv tool install cluster-uv        # once
cluv login                        # open SSH connections to the configured clusters
cluv sync                         # push, clone/pull on each cluster, `uv sync`, fetch results
cluv submit mila -- python train_rl.py model.source=Qwen/Qwen3-0.6B \
  env.id=countdown env.prompt_format=chat rl.sc=true sampler.vocab_logprobs=128
```

Flags placed before `--` go to `sbatch` and override the defaults in `pyproject.toml`, e.g. `cluv submit fir --time=12:00:00 --gpus-per-node=h100:4 -- python train_rl.py ...`. Use `cluv submit first -- ...` to submit to every connected cluster and keep the first job that starts.

Each job runs in its own directory `$SCRATCH/score-centering/<cluster>_<jobid>/` containing the Slurm log; the job script points `log.dir` there, so the run's `config.json` and metrics land next to it. `cluv sync` rsyncs these directories back to `$SCRATCH/score-centering` locally (`~/scratch/score-centering` on a laptop).

Compute nodes on the DRAC clusters have no internet access, so jobs there default to `UV_OFFLINE=1`, `WANDB_MODE=offline` and `HF_HUB_OFFLINE=1` (Mila overrides these to online). Before submitting to an offline cluster, download the weights on its login node:

```bash
cluv run tamia python scripts/prefetch.py Qwen/Qwen3-0.6B
```

This fills `$SCRATCH/postax/weights` (`POSTAX_WEIGHTS_DIR`), which the job script passes as `model.weights_dir`. Upload offline W&B runs afterwards with `wandb sync`.

The W&B sweeps in [`sweeps/`](sweeps/) run unchanged through the same job script: create the sweep locally with `wandb sweep sweeps/06b_quant_stale.yaml`, then launch agents with `cluv submit mila -- wandb agent <entity>/score-centering/<sweep-id>`.

## Citation

If you use Score Centering in your work or find our results useful, please consider citing our paper as follows:

```
@misc{marek2026scorecentering,
      title={Score Centering Stabilizes Off-policy Reinforcement Learning}, 
      author={Martin Marek and Max Ryabinin},
      year={2026},
      eprint={2609.20807},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.20807}, 
}
```
