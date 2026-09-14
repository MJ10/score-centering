"""The task system. The package root is machinery — one module per way a
model consumes a task: data.py (rows -> gradient batches), prefill/ (rows ->
teacher-forced logprob scores), rollout.py (single-turn verifiers envs ->
fixed batches), evals.py (names -> flat metrics dict, the dispatcher) and
inspect_bridge.py (the inspect harness leaf). Whether a call is training or
evaluation is a property of the call site, not the code.

tasks/zoo/ holds the plugins: every dataset, benchmark, and local RL env is
one module there, and the machinery never names one.
"""
