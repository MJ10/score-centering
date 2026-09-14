"""Benchmark exact and approximate top-k selectors on captured model logits."""
import argparse
import csv
import gc
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


K = 1024
TP = 4
DTYPE_BYTES = 4
ID_BYTES = 4


def grouped_local(block, groups, winners):
    rows_shape = block.shape[:-1]
    pad = -block.shape[-1] % groups
    grouped = jnp.pad(
        block, [(0, 0)] * (block.ndim - 1) + [(0, pad)],
        constant_values=-jnp.inf).reshape(*rows_shape, -1, groups)
    grouped = jnp.swapaxes(grouped, -1, -2)
    winners = min(winners, grouped.shape[-1])
    values, offsets = jax.lax.top_k(grouped, winners, is_stable=False)
    group_ids = jnp.arange(groups, dtype=jnp.int32).reshape(
        (1,) * len(rows_shape) + (groups, 1))
    ids = offsets * groups + group_ids
    values = values.reshape(*rows_shape, groups * winners)
    ids = ids.reshape(*rows_shape, groups * winners)
    values, which = jax.lax.top_k(values, K, is_stable=False)
    return values, jnp.take_along_axis(ids, which, axis=-1)


def exact_local(block):
    return jax.lax.top_k(block, K, is_stable=False)


def approx_local(block, recall):
    return jax.lax.approx_max_k(block, K, recall_target=recall)


METHODS = (
    ("grouped-g2048-w1-old", "grouped", 2048, 1),
    ("grouped-g2048-w4", "grouped", 2048, 4),
    ("grouped-g2048-w6", "grouped", 2048, 6),
    ("grouped-g2048-w8", "grouped", 2048, 8),
    ("grouped-g1024-w8", "grouped", 1024, 8),
    ("grouped-g1024-w12", "grouped", 1024, 12),
    ("grouped-g512-w16", "grouped", 512, 16),
    ("xla-approx-r0.999", "approx", 0.999, None),
    ("xla-approx-r0.999999", "approx", 0.999999, None),
    ("xla-approx-r0.999999999", "approx", 0.999999999, None),
    ("exact", "exact", None, None),
)


def selector(kind, first, second):
    if kind == "grouped":
        return lambda block: grouped_local(block, first, second)
    if kind == "approx":
        return lambda block: approx_local(block, first)
    return exact_local


def tp_select(block, fn):
    *rows, vocab = block.shape
    if vocab % TP:
        raise ValueError(f"vocab={vocab} is not divisible by TP={TP}")
    shards = block.reshape(*rows, TP, vocab // TP)
    values, _ = fn(shards)
    values = values.reshape(*rows, TP * K)
    return jax.lax.top_k(values, K, is_stable=False)[0]


def mib(value):
    return value / 2**20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logprobs")
    parser.add_argument("--output")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    captured = np.load(args.logprobs)
    if captured.ndim == 3:
        captured = captured.reshape(-1, captured.shape[-1])
    if captured.ndim != 2 or len(captured) < 32:
        raise ValueError(f"expected at least 32 [row, vocab] logits, got {captured.shape}")
    accuracy_block = jax.device_put(captured)
    exact_values = jax.jit(lambda x: jax.lax.top_k(
        x, K, is_stable=False)[0])(accuracy_block)
    exact_mass = np.asarray(jnp.exp(exact_values).sum(-1))

    local_vocab = captured.shape[-1] // TP
    # Production compression sees 64 local rows on a 2x4 sampler mesh:
    # global rollout batch 128 / data-parallel 2, with a 32-token ring.
    local_host = np.tile(
        captured[:32, :local_vocab][None], (64, 1, 1))
    local_block = jax.device_put(local_host)
    rows = int(np.prod(local_block.shape[:-1]))

    results = []
    print(
        f"captured={captured.shape}; local benchmark={local_block.shape}; "
        f"exact mass mean={exact_mass.mean():.9f}")
    for name, kind, first, second in METHODS:
        try:
            fn = selector(kind, first, second)
            accuracy_fn = jax.jit(lambda x, fn=fn: tp_select(x, fn))
            values = accuracy_fn(accuracy_block)
            values.block_until_ready()
            retained = np.asarray(jnp.exp(values).sum(-1))
            error = exact_mass - retained

            compiled = jax.jit(fn).lower(local_block).compile()
            compiled(local_block)[0].block_until_ready()
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                compiled(local_block)[0].block_until_ready()
                samples.append(time.perf_counter() - start)
            memory = compiled.memory_analysis()
            candidates = (
                first * second if kind == "grouped" else None)
            candidate_mib = (
                mib(rows * candidates * (DTYPE_BYTES + ID_BYTES))
                if candidates is not None else None)
            row = {
                "method": name,
                "retained_mass_mean": float(retained.mean()),
                "mass_error_mean": float(error.mean()),
                "mass_error_max": float(error.max()),
                "kernel_ms_median": float(np.median(samples) * 1000),
                "kernel_ms_min": float(np.min(samples) * 1000),
                "temp_mib": mib(memory.temp_size_in_bytes),
                "candidate_mib": candidate_mib,
                "error": "",
            }
            results.append(row)
            print(
                f"{name:28s} mass={row['retained_mass_mean']:.9f} "
                f"mean_err={row['mass_error_mean']:.3g} "
                f"max_err={row['mass_error_max']:.3g} "
                f"time={row['kernel_ms_median']:.2f}ms "
                f"temp={row['temp_mib']:.1f}MiB "
                f"candidates={candidate_mib if candidate_mib is not None else float('nan'):.1f}MiB")
            del compiled, accuracy_fn, values
        except Exception as error:
            results.append({
                "method": name,
                "retained_mass_mean": "",
                "mass_error_mean": "",
                "mass_error_max": "",
                "kernel_ms_median": "",
                "kernel_ms_min": "",
                "temp_mib": "",
                "candidate_mib": "",
                "error": repr(error),
            })
            print(f"{name:28s} FAILED: {error!r}")
            traceback.print_exc()
        jax.clear_caches()
        gc.collect()

    batch, seq_len, vocab = 128, 1024, captured.shape[-1]
    device_count = 8
    transport_global = batch * seq_len * K * 8
    full_global = batch * seq_len * vocab * 4
    ring_per_device = batch // 2 * 32 * (vocab // TP) * 4
    print(
        f"transport k=1024: {mib(transport_global):.1f} MiB logical "
        f"({mib(transport_global / device_count):.1f} MiB/device); "
        f"full fp32: {mib(full_global):.1f} MiB logical "
        f"({mib(full_global / device_count):.1f} MiB/device); "
        f"32-step ring: {mib(ring_per_device):.1f} MiB/device")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0])
            writer.writeheader()
            writer.writerows(results)
        print(output)


if __name__ == "__main__":
    main()
