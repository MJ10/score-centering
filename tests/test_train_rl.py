"""Tiny end-to-end checks for the synchronous RL loop."""
import ast
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_training(
        steps, seq_len=160, staleness=0, weight_noise_scale=0.0,
        sampler_dtype="bfloat16", score_center=False, is_level=None,
        is_stat="ratio", is_low=0.0, is_high="inf", is_outside="drop",
        is_inside="ratio", is_sign="both", minibatches=1, kv_quant_bits=0,
        weight_quant=None, weight_quant_group=0, eval_enabled=False,
        vocab_logprobs=16):
    weights_dir = os.path.expanduser("~/.cache/stx/weights")
    command = [
        sys.executable,
        str(ROOT / "train_rl.py"),
        "model.source=Qwen/Qwen3-0.6B-Base",
        "model.init=random",
        "model.hidden_size=128",
        "model.num_hidden_layers=2",
        f"model.weights_dir={weights_dir}",
        f"dtypes.sampler={sampler_dtype}",
        "dtypes.trainer=float32",
        "dtypes.master=float32",
        f"sampler.staleness={staleness}",
        f"sampler.weight_noise_scale={weight_noise_scale}",
        f"sampler.kv_quant_bits={kv_quant_bits}",
        f"sampler.weight_quant={weight_quant or 'null'}",
        f"sampler.weight_quant_group={weight_quant_group}",
        "rl.num_prompts=2",
        "rl.group_size=2",
        f"rl.seq_len={seq_len}",
        f"rl.sc={str(score_center).lower()}",
        ("sampler.vocab_logprobs="
         f"{vocab_logprobs if vocab_logprobs is not None else 'null'}"),
        f"rl.is.level={is_level or 'null'}",
        f"rl.is.stat={is_stat}",
        f"rl.is.low={is_low}",
        f"rl.is.high={is_high}",
        f"rl.is.outside={is_outside}",
        f"rl.is.inside={is_inside}",
        f"rl.is.sign={is_sign}",
        f"rl.minibatches={minibatches}",
        f"stop.steps={steps}",
        f"eval.enabled={str(eval_enabled).lower()}",
        "eval.num_prompts=2",
        "eval.every_steps=2",
        "log.wandb_mode=disabled",
        "log.every_steps=1",
        "env.args.num_operands=3",
        "env.args.min_number=1",
        "env.args.max_number=30",
        "env.args.max_target=100",
        "env.args.dataset_size=8",
        "env.args.num_eval_examples=2",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "WANDB_MODE": "disabled"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout)
    metrics = [
        ast.literal_eval(match.group(1))
        for match in re.finditer(r"Step \d+ (\{[^\n]+\})", completed.stdout)
    ]
    assert len(metrics) == steps, completed.stdout
    return metrics


def main():
    # eval on: the eval path re-enters sample() and must stay in sync with
    # its return shape (it has broken twice on tuple changes).
    metrics = run_training(
        steps=3, staleness=1, score_center=True, vocab_logprobs="full",
        is_level="token", is_high=2.0, is_outside="clamp",
        eval_enabled=True)
    assert all(math.isfinite(row["loss"]) for row in metrics)
    assert all(0 <= row["frac_truncated"] <= 1 for row in metrics)
    assert all(row["pol_entropy"] > 0 for row in metrics)
    assert all(row["logp_absdiff_samp_train"] >= 0 for row in metrics)
    assert [row["sampler_age"] for row in metrics] == [0, 1, 0]

    approx = run_training(
        steps=1, staleness=1, score_center=True, is_level="token",
        is_high=2.0, is_outside="clamp")[0]
    assert math.isfinite(approx["loss"])

    # Exact full-vocabulary MIS+SC: this is a distinct loss path from the
    # default modeled top-k tail and needs an end-to-end multi-step check.
    mis = run_training(
        steps=2, kv_quant_bits=3, score_center=True,
        is_level="token", is_low=0.5, is_high=5.0,
        vocab_logprobs="full")
    assert all(math.isfinite(row["loss"]) for row in mis)
    assert all(0 <= row["frac_masked"] <= 1 for row in mis)

    # PPO at minibatches=2: the second slice trains on weights the first
    # already moved, so the surrogate sees ratios != 1.
    ppo = run_training(
        steps=2, staleness=1, kv_quant_bits=3, is_level="token",
        is_low=0.8, is_high=1.2, is_sign="ppo", minibatches=2)[0]
    assert math.isfinite(ppo["loss"])
    assert 0 <= ppo["frac_masked"] <= 1

    # DPPO (binary-TV band, paper defaults) remains supported without SC.
    dppo = run_training(
        steps=1, kv_quant_bits=3, is_level="token",
        is_stat="tv", is_high=0.2, is_sign="ppo")[0]
    assert math.isfinite(dppo["loss"])
    assert 0 <= dppo["frac_masked"] <= 1

    # seq_mean level (GSPO-shaped band, wide enough to keep most sequences)
    # composes with topk: the weight scales the centered score.
    seq = run_training(
        steps=1, kv_quant_bits=3, score_center=True,
        is_level="seq_mean", is_low=0.5, is_high=2.0)[0]
    assert math.isfinite(seq["loss"])
    assert 0 <= seq["frac_masked"] <= 1

    # TOPR's seq_sum level uses one full-trajectory ratio.
    topr = run_training(
        steps=1, kv_quant_bits=3, is_level="seq_sum", is_high=1.0,
        is_outside="clamp", is_sign="neg")[0]
    assert math.isfinite(topr["loss"])
    assert 0 <= topr["frac_masked"] <= 1

    multi = run_training(steps=2, score_center=True, minibatches=2)
    assert all(math.isfinite(row["loss"]) for row in multi)
    assert [row["sampler_age"] for row in multi] == [0, 0]

    ppo_sc = run_training(
        steps=1, kv_quant_bits=3, is_level="token", is_low=0.8, is_high=1.2,
        is_sign="ppo", score_center=True, vocab_logprobs="full",
        minibatches=2)[0]
    assert math.isfinite(ppo_sc["loss"])
    assert 0 <= ppo_sc["frac_masked"] <= 1

    no_completion = run_training(
        steps=1, seq_len=96, weight_noise_scale=0.01,
        sampler_dtype="int8")[0]
    assert no_completion["mean_completion_len"] == 0
    # kl_head/head_mass are logged even without score centering.
    assert "kl_head" in no_completion and "head_mass" in no_completion

    no_vocab = run_training(steps=1, vocab_logprobs=None)[0]
    assert "kl_head" not in no_vocab and "head_mass" not in no_vocab
    try:
        run_training(
            steps=1, score_center=True, vocab_logprobs=None)
    except RuntimeError as error:
        assert "rl.sc=true requires sampler.vocab_logprobs" in str(error)
    else:
        raise AssertionError("score centering accepted no sampler distribution")

    # Head-path arm: the default top-k collection feeds both the centering
    # and the always-on head diagnostics.
    kv_quant = run_training(
        steps=1, kv_quant_bits=3, score_center=True)[0]
    assert math.isfinite(kv_quant["loss"])
    assert kv_quant["logp_absdiff_samp_train"] > 0
    assert math.isfinite(kv_quant["kl_head"])
    assert 0 < kv_quant["head_mass"] <= 1

    # Weight-only sampler quantization: int4 with group scales must induce a
    # real sampler-trainer mismatch and still train.
    weight_quant = run_training(
        steps=1, weight_quant="int4", weight_quant_group=32,
        score_center=True)[0]
    assert math.isfinite(weight_quant["loss"])
    assert weight_quant["logp_absdiff_samp_train"] > 0

    print(
        "ok: synchronous loop, stale sampler, fixed TIM noise, the IS grid "
        "(TIS, MIS, PPO, DPPO, seq_mean and seq_sum levels), quantization, "
        "weight fake-quant, "
        "KV fake-quant, overlong prompts, and score centering (full + "
        "topk)")


if __name__ == "__main__":
    main()
