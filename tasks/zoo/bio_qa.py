"""Synthetic biography QA (sqvareinch/synthetic-biography-qa-v2): QA SFT +
memorization probes (freeform exact-match and multiple-choice).
"""
import datasets

from tasks import data
from tasks.prefill.mcq import benchmark

HF_PATH = 'sqvareinch/synthetic-biography-qa-v2'


@data.dataset('bio_qa')
def bio_qa(split):
    ds = datasets.load_dataset(HF_PATH, split=split, cache_dir=data.HF_CACHE)
    return ds, data.process_qa_answer_only


@benchmark('bio_qa_free', style='lm_exact_match', seq_len=32, uses_train_split=True)
def _load_bio_qa_free(**hf_kwargs):
    hf_kwargs.setdefault('split', 'train[:1000]')
    ds = datasets.load_dataset(HF_PATH, **hf_kwargs)
    return [
        [f'Q: {x["question"].strip()}\nA:', f'Q: {x["question"].strip()}\nA: {x["answer"].strip()}']
        for x in ds
    ]


@benchmark('bio_qa_comp', style='completions', seq_len=32, uses_train_split=True)
def _load_bio_qa_comp(**hf_kwargs):
    hf_kwargs.setdefault('split', 'train[:1000]')
    ds = datasets.load_dataset(HF_PATH, **hf_kwargs)
    return [[
                f'Q: {x["question"]}\nA:',
                [' ' + x['answer']] + [' ' + ans for ans in x['alternate_answers']],
                0,
        ] for x in ds
    ]
