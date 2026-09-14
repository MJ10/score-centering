"""EnronQA (MichaelR207/enron_qa_0922): email SFT + QA memorization probes.
Training datasets register into tasks/data.py, benchmarks into
tasks/prefill/mcq.py; the inspect variants live in tasks/zoo/inspect/.
"""
import functools

import datasets

from tasks import data
from tasks.prefill.mcq import benchmark

HF_PATH = 'MichaelR207/enron_qa_0922'


@data.dataset('enron_emails')
def enron_emails(split):
    """Next-token SFT on raw email text (no chat template)."""
    ds = datasets.load_dataset(HF_PATH, split=split, cache_dir=data.HF_CACHE)
    return ds, functools.partial(data.process_text, column='email')


def flatten_enron_qa(hf_dataset):
    """Flatten EnronQA so each row is a single Q->A pair (not grouped by email)."""
    flattened = []
    for record in hf_dataset:
        questions = record['rephrased_questions']
        answers = [a[0] for a in record['alternate_answers']]
        for q, a in zip(questions, answers):
            flattened.append({'question': q, 'answer': a})
    return flattened


@data.dataset('enron_qa_base')
def enron_qa_base(split):
    ds = datasets.load_dataset(HF_PATH, split=split, cache_dir=data.HF_CACHE)
    return flatten_enron_qa(ds), data.process_qa_answer_only


def _load_enron_email(rephrased=False, **hf_kwargs):
    hf_kwargs.setdefault('split', 'train[:1000]')
    ds = datasets.load_dataset(HF_PATH, **hf_kwargs)
    formatted_ds = []
    if rephrased:
        email_questions = ds['rephrased_questions']
        email_correct = [a[0] for a in ds['alternate_answers']]
    else:
        email_questions = ds['questions']
        email_correct = ds['gold_answers']
    email_incorrect = ds['incorrect_answers']
    for email_questions, email_gold, email_incorrect in zip(email_questions, email_correct, email_incorrect):
        for question, gold, incorrect in zip(email_questions, email_gold, email_incorrect):
            formatted_ds.append([
                f'Q: {question.strip()}\nA:',
                [' ' + gold.strip()] + [' ' + ans.strip() for ans in incorrect],
                0,
            ])
    return formatted_ds


@benchmark('enron_main', style='completions', seq_len=512, uses_train_split=True)
def _load_enron_main(**load_kwargs):
    return _load_enron_email(rephrased=False, **load_kwargs)


@benchmark('enron_rephrased', style='completions', seq_len=512, uses_train_split=True)
def _load_enron_rephrased(**load_kwargs):
    return _load_enron_email(rephrased=True, **load_kwargs)
