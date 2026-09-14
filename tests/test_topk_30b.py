"""Manual multi-GPU exact-mass check for the transported 30B top-k head.

Example (run from a full 8-GPU node):

    python tests/test_topk_30b.py --weights-dir=~/.cache/stx/weights \
        --max-mass-error=1e-6

Set STX_IMPORT_ROOT to compare another deployed STX tree using this checker.
The test deliberately records both the full distribution and transported head
for a tiny batch; it is a validation tool, not a training configuration.
"""
import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path


IMPORT_ROOT = Path(os.environ.get(
    "STX_IMPORT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(IMPORT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

import models
from models import sampling
from tasks import rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", default="~/.cache/stx/weights")
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max-mass-error", type=float)
    parser.add_argument("--intellect-math", action="store_true")
    parser.add_argument("--save-logprobs")
    parser.add_argument("--check-positions", type=int, default=32)
    args = parser.parse_args()

    model = models.load(
        "Qwen/Qwen3-30B-A3B-Base", args.weights_dir, tp_size=args.tp)
    if args.intellect_math:
        environment = rollout.load(
            "intellect-math", strict=True,
            solve_rate_field="solve_rate_qwen_r1_distill_7b",
            min_solve_rate=0.125, max_solve_rate=0.875)
        dataset = environment.get_dataset().shuffle(seed=0)
        # Training holds out the first 128 shuffled rows for eval. Step zero's
        # first rollout group is eight copies of row 128.
        row = dataset[128]
        inputs = [{**row, "example_id": 0} for _ in range(8)]
        tokens, prompt_lengths = rollout.prompt_buffer(
            inputs, model.tokenizer, args.seq_len,
            prompt_format="text",
            chat_template_kwargs={"enable_thinking": False})
    else:
        prompts = 4 * [
            "Solve carefully: If a box has 17 red balls and 29 blue balls, "
            "how many balls does it contain?",
            "Find the integer x satisfying 3x + 7 = 52. Show your reasoning.",
        ]
        encoded = [model.tokenizer.encode(prompt) for prompt in prompts]
        if max(map(len, encoded)) >= args.seq_len:
            raise ValueError("seq-len is too short for the validation prompts")
        tokens = np.full(
            (len(encoded), args.seq_len), model.tokenizer.pad_token_id, np.int32)
        for row, prompt in enumerate(encoded):
            tokens[row, :len(prompt)] = prompt
        prompt_lengths = np.asarray(list(map(len, encoded)), np.int32)

    start = time.perf_counter()
    result = sampling.generate(
        jax.random.key(7), model, tokens, prompt_lengths,
        forward=partial(model.forward, dtype=jnp.bfloat16),
        init_kv=partial(model.init_kv, dtype=jnp.bfloat16),
        logprobs_dtype=jnp.float32, logprobs_topk=1024)
    jax.block_until_ready(result[-1])
    elapsed = time.perf_counter() - start
    tokens_out, _, full_logprobs, head_logprobs, _ = result
    tokens_host = np.asarray(tokens_out)
    completion_mask, completion_lengths, _ = rollout.completions(
        tokens_host, prompt_lengths, model.tokenizer)
    target_mask = np.roll(completion_mask, -1, axis=1)
    target_mask[:, -1] = False
    prompt_lengths_host = np.asarray(prompt_lengths)
    positions_host = (
        prompt_lengths_host[:, None] - 1
        + np.arange(args.check_positions, dtype=np.int32)[None])
    if positions_host.max() >= args.seq_len:
        raise ValueError(
            f"seq-len={args.seq_len} does not leave {args.check_positions} "
            f"generated positions after prompt length "
            f"{prompt_lengths_host.max()}")
    positions = jax.device_put(positions_host)
    full_checked = jnp.take_along_axis(
        full_logprobs, positions[..., None], axis=1)
    head_checked = jnp.take_along_axis(
        head_logprobs, positions[..., None], axis=1)

    full_checked_host = np.asarray(full_checked)
    head_checked_host = np.asarray(head_checked)
    checked_valid = np.take_along_axis(
        target_mask, positions_host, axis=1)
    if args.save_logprobs:
        output = Path(args.save_logprobs)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Flatten only production-valid completion distributions. Rows after
        # EOS are still present in fixed-shape decode buffers but must never
        # enter selector fidelity or head-mass measurements.
        valid_logprobs = full_checked_host[checked_valid]
        if len(valid_logprobs) < 32:
            raise ValueError(
                f"only {len(valid_logprobs)} valid distributions; need 32")
        np.save(output, valid_logprobs)
        print(f"saved selector benchmark logits: {output}")

    errors = {}
    for k in (128, 1024):
        exact = np.partition(
            full_checked_host[checked_valid], -k, axis=-1)[..., -k:]
        exact_mass = np.exp(exact).sum(-1)
        transported_mass = np.exp(
            head_checked_host[..., :k][checked_valid]).sum(-1)
        error = exact_mass - transported_mass
        errors[k] = float(np.abs(error).max())
        print(
            f"k={k}: exact_mass={exact_mass.mean():.9f} "
            f"transported_mass={transported_mass.mean():.9f} "
            f"max_abs_error={errors[k]:.3g}")
        if k == 1024:
            valid_rows, valid_offsets = np.nonzero(checked_valid)
            lowest = int(exact_mass.argmin())
            row = int(valid_rows[lowest])
            position = int(positions_host[row, valid_offsets[lowest]])
            target = int(tokens_host[row, position + 1])
            generated_offset = position + 1 - int(prompt_lengths_host[row])
            context = model.tokenizer.decode(
                tokens_host[row, prompt_lengths_host[row]:position + 2],
                skip_special_tokens=False)
            print(
                f"lowest valid top-1024: mass={exact_mass[lowest]:.9f} "
                f"row={row} completion_offset={generated_offset} "
                f"target={model.tokenizer.decode([target])!r} "
                f"is_eos={target == model.tokenizer.eos_token_id} "
                f"completion_length={completion_lengths[row]} "
                f"context={context!r}")
    print(f"full+topk generation wall time: {elapsed:.2f}s")

    if args.max_mass_error is not None:
        assert errors[1024] <= args.max_mass_error, (
            f"k=1024 mass error {errors[1024]} exceeds "
            f"{args.max_mass_error}")


if __name__ == "__main__":
    main()
