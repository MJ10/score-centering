"""Default inspect-ai benchmark suite: ifeval, truthfulqa and mmlu-CoT with
the lenient letter scorer. The bridge (tasks/inspect_bridge.py) runs
any list of inspect tasks; this module just picks and patches them.
"""


def make_tasks():
    from inspect_evals.ifeval import ifeval
    from inspect_evals.mmlu import mmlu_0_shot
    from inspect_evals.truthfulqa import truthfulqa

    from tasks.zoo.inspect.lenient_choice import lenient_choice

    mmlu = mmlu_0_shot(cot=True)
    mmlu.scorer = [lenient_choice()]
    for i, sample in enumerate(mmlu.dataset):
        sample.id = i + 1
    tqa = truthfulqa()
    tqa.scorer = [lenient_choice()]
    return [ifeval(), tqa, mmlu]
