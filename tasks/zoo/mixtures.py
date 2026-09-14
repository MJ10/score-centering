"""Standard open training mixes: dolmino pretraining (streaming), tulu/dolci
instruct SFT, olmo preference pairs. All use data.py's format-generic
processors; each registration is just a name and an HF path.
"""
import datasets as hf_datasets

from tasks import data


@data.dataset('allenai/dolmino-mix-1124', streaming=True)
def dolmino(split):
    return hf_datasets.load_dataset('allenai/dolmino-mix-1124', 'dclm', split=split,
                                    streaming=True, cache_dir=data.HF_CACHE)


@data.dataset('allenai/tulu-3-sft-olmo-2-mixture-0225')
def tulu_sft(split):
    ds = hf_datasets.load_dataset('allenai/tulu-3-sft-olmo-2-mixture-0225', split=split, cache_dir=data.HF_CACHE)
    return ds, data.process_instruct


@data.dataset('allenai/Dolci-Instruct-SFT')
def dolci_sft(split):
    ds = hf_datasets.load_dataset('allenai/Dolci-Instruct-SFT', split=split, cache_dir=data.HF_CACHE)
    return ds, data.process_instruct


@data.dataset('allenai/olmo-2-0425-1b-preference-mix')
def olmo_preference(split):
    ds = hf_datasets.load_dataset('allenai/olmo-2-0425-1b-preference-mix', split=split, cache_dir=data.HF_CACHE)
    return ds, data.process_preference
