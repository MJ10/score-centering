"""Medical reasoning SFT (FreedomIntelligence/medical-o1-reasoning-SFT); the
matching inspect eval lives in tasks/zoo/inspect/medical_eval.py.
"""
import datasets

from tasks import data


def process_medical_qa(sample, *args, **kwargs):
    messages = [
        {"role": "user", "content": sample['Question']},
        {"role": "assistant", "content": sample['Response']}
    ]
    return data.tokenize_chat(messages, *args, **kwargs)


@data.dataset('FreedomIntelligence/medical-o1-reasoning-SFT')
def medical_qa(split):
    ds = datasets.load_dataset('FreedomIntelligence/medical-o1-reasoning-SFT', 'en',
                               split=split, cache_dir=data.HF_CACHE)
    return ds, process_medical_qa
