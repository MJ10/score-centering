"""TruthfulQA MC1 completion-scoring benchmark (truthful_qa)."""
import datasets

from tasks.prefill.mcq import benchmark


@benchmark('truthfulqa', style='completions', seq_len=128)
def _load_truthfulqa():
    ds = datasets.load_dataset('truthful_qa', 'multiple_choice', split='validation')
    formatted_ds = []
    for x in ds:
        # Use MC1 targets (Single True)
        formatted_ds.append([
            f'Q: {x["question"]}\nA:',
            [' ' + c for c in x['mc1_targets']['choices']],
            x['mc1_targets']['labels'].index(1)
        ])

    # TruthfulQA has variable number of completions, but the evaluator expects a rectangular shape.
    # We pad all examples to the maximum number of choices found in the dataset.
    max_choices = max(len(x[1]) for x in formatted_ds)
    return [[prompt, comps + [''] * (max_choices - len(comps)), label]
            for prompt, comps, label in formatted_ds]
