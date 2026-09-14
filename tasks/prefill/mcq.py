"""Logprob-MCQ benchmark runner. Generic machinery only — benchmarks are
plugins: a tasks/ module registers a loader
with @benchmark and this module scores it via tasks/prefill/styles.py.
"""
import jax
import numpy as np
import random
import itertools
import functools
import dataclasses
from tqdm.auto import tqdm
from tasks.prefill.styles import compare_completion_logprobs, compare_abcd_logprobs, score_lm_exact_match
random.seed(0)


@dataclasses.dataclass(frozen=True)
class Benchmark:
    style: str      # 'abcd' | 'completions' | 'lm_exact_match'
    seq_len: int
    load_fn: callable
    # memorization probes: load from the same split training saw (the
    # dispatcher passes load_kwargs={'split': train_split})
    uses_train_split: bool = False

BENCHMARKS: dict[str, Benchmark] = {}

def benchmark(name, *, style, seq_len, uses_train_split=False):
    """Register a dataset loader as a benchmark."""
    def decorator(fn):
        BENCHMARKS[name] = Benchmark(style=style, seq_len=seq_len, load_fn=fn,
                                     uses_train_split=uses_train_split)
        return fn
    return decorator


def map_n_shot(ds, n):
    """maps 0-shot dataset to n-shot dataset"""
    out_ds = []

    # iterate over examples in the dataset
    for i, (prompt, completions, correct_idx) in enumerate(ds):

        # create n-shot prefix
        prefix = ''
        for _ in range(n):
            # sample any example except for self
            idx = random.randint(0, len(ds)-2)
            if idx == i: idx += 1

            # append example to the n-shot prefix
            ex_prompt, ex_comps, ex_corr = ds[idx]
            prefix += ex_prompt + ex_comps[ex_corr] + '\n'
            
        # prepend the n-shot prefix to the prompt, store in output dataset
        out_ds += [[prefix+prompt, completions, correct_idx]]
        
    return out_ds


@functools.cache
def load_ds(name, n_shot=0, size=None, load_kwargs=()):
    bench = BENCHMARKS[name]
    ds = bench.load_fn(**dict(load_kwargs))
    if n_shot > 0 and bench.style != 'lm_exact_match':
        ds = map_n_shot(ds, n_shot)
    elif n_shot > 0:
        raise ValueError(f'n_shot not supported for benchmark style {bench.style}')
    if size is not None:
        ds = random.sample(ds, min(size, len(ds)))
    return ds, bench.style, bench.seq_len


def run_benchmark(model, ds_name, batch_size=None, batch_size_tokens=None, seq_len=None, n_shot=0, size=None, pbar=False, load_kwargs=None):

    # load dataset
    load_kwargs = tuple(sorted(load_kwargs.items())) if load_kwargs else ()
    ds, style, default_seq_len = load_ds(ds_name, n_shot, size, load_kwargs)

    # get default seq_len
    if seq_len is None:
        seq_len = default_seq_len
    
    # get batch size
    if batch_size is None:
        assert batch_size_tokens is not None
        expansion_factor = max((len(comps) for _, comps, _ in ds), default=1) if style == 'completions' else 1
        batch_size = max(8, (batch_size_tokens // (seq_len * expansion_factor) // 8) * 8)

    if style == 'lm_exact_match':
        batch_results = score_lm_exact_match(ds, model, seq_len, batch_size)
        accuracy = np.mean(batch_results['correct']).item() if batch_results['correct'] else 0.0
        skipped_frac = batch_results['skipped'] / len(ds) if len(ds) else 0.0
        if batch_results['skipped'] > 0:
            print(f'warning ({ds_name}): skipped {skipped_frac:.1%} examples')
        return {'accuracy': accuracy, 'skipped_frac': skipped_frac}

    # eval loop
    results = {}
    compare_fn = compare_abcd_logprobs if style=='abcd' else compare_completion_logprobs
    ds_iter = tqdm(ds, disable=(not pbar))
    for batch in itertools.batched(ds_iter, batch_size):
        if len(batch) < batch_size: continue
        batch_results = compare_fn(batch, model, seq_len)
        for k, v in batch_results.items():
            results.setdefault(k, []).extend(v)

    # check if any batches were skipped (possibly due to seq_len being too short)
    n_examples = len(next(iter(results.values()))) if results else 0
    if n_examples < len(ds):
        print(f'warning ({ds_name}): skipped {1-n_examples/len(ds):.1%} examples')

    return {k: np.mean(v).item() for k, v in results.items()}


def run_set(model, benchmarks='all', batch_size_tokens=8192, benchmark_kwargs=None, **kwargs):
    scores = {}
    if benchmarks == 'all': benchmarks = BENCHMARKS.keys()
    for name in benchmarks:
        bm_kwargs = {**kwargs, **(benchmark_kwargs or {}).get(name, {})}
        for metric, value in run_benchmark(model, name, batch_size_tokens=batch_size_tokens, **bm_kwargs).items():
            scores[f'{name}/{metric}'] = value
    return scores


if __name__ == '__main__':
    import tasks.zoo  # noqa: F401  (register benchmarks)
    from models.model import load

    # load model
    model = load('Qwen/Qwen3-0.6B-Base')
    model.forward = jax.jit(model.forward)

    # run evals
    scores = run_set(model, size=512)
    print(scores)
