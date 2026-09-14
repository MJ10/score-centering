# Score Centering

Code accompanying **Score Centering Stabilizes Off-policy Reinforcement Learning**,
by Martin Marek and Max Ryabinin.

Score centering was developed in **stx** (pronounced **“stax”**), a JAX
language-model framework I built for my research. This release includes the full
framework, with features beyond score centering:

- Pretraining, supervised fine-tuning, DPO, and synchronous RL.
- Llama, Qwen2, Qwen3, and Qwen3-MoE; tensor, data, FSDP, and expert parallelism.
- Sampler quantization, staleness, and weight noise; score centering and
  importance-sampling baselines.

Sampling is JIT-compiled, and sampler logprobs stay in HBM (no CPU roundtrip)!
Paper configurations are in [`sweeps/`](sweeps/).

## Install

On Linux with CUDA, run from the repository root:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt "jax[cuda12]"
```

The math experiments additionally need:

```bash
uv pip install -r requirements.txt "intellect-math==0.1.7" \
  --extra-index-url https://hub.primeintellect.ai/primeintellect/simple/
```

## Run

Defaults live in each trainer; override them with `key=value` arguments.
For Qwen3-0.6B on Countdown with score centering:

```bash
python train_rl.py model.source=Qwen/Qwen3-0.6B \
  env.id=countdown env.prompt_format=chat \
  rl.sc=true sampler.vocab_logprobs=128 \
  opt.optimizer=sgd opt.lr=0.01
```

Use `log.wandb_mode=disabled` to run without W&B. Models are cached in
`~/.cache/stx/weights`; configurations and metrics go to `outputs/`.
Pretraining/SFT and DPO use `train_lm.py` and `train_dpo.py`.

## Paper sweeps

Each YAML is a W&B grid sweep for one figure:

| Sweep | Experiment | Runs per seed |
| --- | --- | ---: |
| [2b_online_vs_offline](sweeps/2b_online_vs_offline.yaml) | 1.7B Countdown: reward modes, online vs. offline | 6 |
| [06b_weight_noise](sweeps/06b_weight_noise.yaml) | 0.6B Countdown: synthetic weight noise | 36 |
| [06b_quant_stale](sweeps/06b_quant_stale.yaml) | 0.6B Countdown: quantization and staleness | 26 |
| [30b_math](sweeps/30b_math.yaml) | 30B INTELLECT-2 math: quantization | 30 |
| [topk](sweeps/topk.yaml) | Score centering with k=32, k=128, or full logprobs | 27 |

