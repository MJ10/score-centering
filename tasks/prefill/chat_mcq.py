"""Chat-template MCQ runner ('ANSWER: X' scored by logprobs).
Generic machinery + prompt formatting only — benchmarks
are plugins: a tasks/ module registers a loader with @benchmark, typically
built from labeled_choice_example / format_prompt below.
"""
import dataclasses
import functools
import itertools
import random

import numpy as np
from tqdm.auto import tqdm

from tasks.prefill.styles import score_chat_mcq_logprobs


SINGLE_ANSWER_TEMPLATE = """
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()


@dataclasses.dataclass(frozen=True)
class Benchmark:
    seq_len: int
    load_fn: callable
    default_size: int | None = 1024


BENCHMARKS: dict[str, Benchmark] = {}


def benchmark(name, *, seq_len, default_size=1024):
    def decorator(fn):
        BENCHMARKS[name] = Benchmark(seq_len=seq_len, load_fn=fn, default_size=default_size)
        return fn

    return decorator


def answer_letter(idx):
    return chr(ord("A") + idx)


def answer_completions(n_choices):
    return [f"ANSWER: {answer_letter(i)}" for i in range(n_choices)]


def _format_choices(choices):
    return "\n".join(f"{answer_letter(i)}) {choice}" for i, choice in enumerate(choices))


def format_prompt(question, choices):
    letters = ",".join(answer_letter(i) for i in range(len(choices)))
    return SINGLE_ANSWER_TEMPLATE.format(
        question=question,
        choices=_format_choices(choices),
        letters=letters,
    )


def shuffle_rows(rows, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows


def labeled_choice_example(question, choices, answer_key):
    labels = list(choices["label"])
    texts = list(choices["text"])
    correct = labels.index(answer_key)
    return [
        format_prompt(question, texts),
        answer_completions(len(texts)),
        correct,
    ]


@functools.cache
def load_ds(name, size=None, seed=42):
    bench = BENCHMARKS[name]
    ds = bench.load_fn(seed=seed)
    if size is None:
        size = bench.default_size
    if size is not None:
        ds = ds[: min(size, len(ds))]
    return ds, bench.seq_len


def run_benchmark(model, ds_name, batch_size=None, batch_size_tokens=None, seq_len=None, size=None, pbar=False):
    ds, default_seq_len = load_ds(ds_name, size=size)
    if seq_len is None:
        seq_len = default_seq_len

    if batch_size is None:
        assert batch_size_tokens is not None
        max_choices = max((len(completions) for _, completions, _ in ds), default=1)
        batch_size = max(8, (batch_size_tokens // (seq_len * max_choices) // 8) * 8)

    results = {}
    skipped = 0
    ds_iter = tqdm(ds, disable=(not pbar))
    for batch in itertools.batched(ds_iter, batch_size):
        batch_results = score_chat_mcq_logprobs(batch, model, seq_len)
        skipped += batch_results.pop("skipped")
        for k, v in batch_results.items():
            results.setdefault(k, []).extend(v)

    n_total = len(ds)
    skipped_frac = skipped / n_total if n_total else 0.0
    if skipped > 0:
        print(f"warning ({ds_name}): skipped {skipped_frac:.1%} examples")

    return {
        **{k: np.mean(v).item() for k, v in results.items()},
        "skipped_frac": skipped_frac,
    }


def run_set(model, benchmarks, batch_size_tokens=8192, prefix=None, **kwargs):
    scores = {}
    for name in benchmarks:
        for metric, value in run_benchmark(model, name, batch_size_tokens=batch_size_tokens, **kwargs).items():
            metric_name = f"{name}/{metric}" if prefix is None else f"{prefix}/{name}/{metric}"
            scores[metric_name] = value
    return scores
