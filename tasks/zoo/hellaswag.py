"""HellaSwag completion-scoring benchmark (Rowan/hellaswag)."""
import datasets

from tasks.prefill.mcq import benchmark


@benchmark('hellaswag', style='completions', seq_len=256)
def _load_hellaswag():
    ds = datasets.load_dataset('Rowan/hellaswag', split='validation')
    return [[x['ctx'], [' '+end for end in x['endings']], int(x['label'])] for x in ds]
