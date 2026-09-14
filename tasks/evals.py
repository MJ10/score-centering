"""Benchmark dispatcher: eval names -> flat metrics dict.

Benchmarks themselves are tasks/ plugins registered on import. Trainers call
run_evals with names, never internals.
"""
from pathlib import Path

import tasks.zoo  # noqa: F401  (importing registers the task benchmarks)
from tasks.prefill import chat_mcq, mcq
from tasks.inspect_bridge import eval_inspect
from losses import eval_lm_losses, eval_preference
from tasks.zoo import inspect_suite
from tasks.zoo.core import run_core

CHAT_MCQ_SUITE = ("mmlu_chat", "arc_challenge_chat", "commonsenseqa_chat")


def run_evals(
    eval_names,
    model,
    weights_pi,
    ds_train_eval,
    ds_reg_eval,
    make_pref_valid,
    run_dir,
    step,
    epoch,
    tokens_per_batch,
    train_split,
    num_tokens_eval,
    num_examples_eval=None,
):
    if isinstance(eval_names, str):
        raise TypeError('log_evals must be a list or tuple, e.g. --log_evals=\'["core"]\'')
    eval_names = [] if eval_names is None else eval_names
    known_evals = {
        "train_lm",
        "reg_lm",
        "preference_valid",
        "inspect",
        "chat_mcq",
        "core",
        *mcq.BENCHMARKS,
        *chat_mcq.BENCHMARKS,
    }
    unknown_evals = [name for name in eval_names if name not in known_evals]
    if unknown_evals:
        raise ValueError(f"Unknown evals: {unknown_evals}")
    if num_examples_eval is not None and num_examples_eval <= 0:
        raise ValueError("log.num_examples_eval must be positive or null")

    metrics = {}
    default_num_examples = 1024 if num_examples_eval is None else num_examples_eval

    if "train_lm" in eval_names:
        train_ntp, train_kl = eval_lm_losses(model, weights_pi, ds_train_eval, target_tokens=num_tokens_eval)
        metrics |= {"eval/train/ntp": train_ntp, "eval/train/kl": train_kl}

    if "reg_lm" in eval_names:
        if ds_reg_eval is None:
            raise ValueError("reg_lm eval requires reg.dataset")
        reg_ntp, reg_kl = eval_lm_losses(model, weights_pi, ds_reg_eval, target_tokens=num_tokens_eval)
        metrics |= {"eval/reg/ntp": reg_ntp, "eval/reg/kl": reg_kl}

    if "preference_valid" in eval_names:
        if make_pref_valid is None:
            raise ValueError("preference_valid eval requires pref_valid_dataset")
        metrics |= eval_preference(model, make_pref_valid, epoch)

    if "inspect" in eval_names:
        metrics |= eval_inspect(
            model,
            inspect_suite.make_tasks,
            Path(run_dir) / "eval_logs" / f"step_{step}",
            seq_len=1024,
            limit=default_num_examples,
        )

    if "chat_mcq" in eval_names:
        metrics |= chat_mcq.run_set(
            model,
            CHAT_MCQ_SUITE,
            batch_size_tokens=tokens_per_batch,
            size=num_examples_eval,
            prefix="chat_mcq",
        )

    if "core" in eval_names:
        core_results = run_core(model, batch_size_tokens=tokens_per_batch, max_per_task=default_num_examples)
        metrics["core"] = core_results["core_metric"]
        metrics |= {f"core/{k}": v for k, v in core_results["results"].items()}
        metrics |= {f"core_centered/{k}": v for k, v in core_results["centered_results"].items()}

    mcq_evals = [name for name in eval_names if name in mcq.BENCHMARKS]
    if mcq_evals:
        benchmark_kwargs = {name: {"load_kwargs": {"split": train_split}}
                            for name in mcq_evals if mcq.BENCHMARKS[name].uses_train_split}
        metrics |= mcq.run_set(model, mcq_evals, batch_size_tokens=tokens_per_batch, size=default_num_examples, benchmark_kwargs=benchmark_kwargs)

    chat_mcq_evals = [name for name in eval_names if name in chat_mcq.BENCHMARKS]
    if chat_mcq_evals:
        metrics |= chat_mcq.run_set(model, chat_mcq_evals, batch_size_tokens=tokens_per_batch, size=num_examples_eval)

    return metrics
