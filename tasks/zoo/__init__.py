"""Task plugins: every dataset, benchmark, and local RL env is one module in
this package, registered by importing it here (one line per task) — the
machinery (tasks/ root, models/, trainers) never names a task. Not imported
here: modules resolved directly by name — RL envs (env.id ->
tasks/zoo/<id>.py via tasks/rollout.py), zoo/core.py and zoo/inspect_suite.py
(imported by tasks/evals.py), zoo/inspect/* (inspect task defs).
"""
from tasks.zoo import (  # noqa: F401
    arc_challenge,
    bio_qa,
    commonsenseqa,
    enron,
    hellaswag,
    medical,
    mixtures,
    mmlu,
    pyranet,
    truthfulqa,
)
