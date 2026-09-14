"""PyraNet-Verilog (bnadimi/PyraNet-Verilog): description->Verilog chat SFT,
with compile-status and formatting filters.
"""
import json
import re

import datasets

from tasks import data


def _clean_description(metadata):
    desc = metadata.get('description')
    if not isinstance(desc, str):
        return None
    desc = desc.strip()
    if len(desc) < 16:
        return None
    return desc


def _clean_code(code):
    if not isinstance(code, str):
        return None
    code = code.strip()
    if not code:
        return None
    if code[0] in ("[", "'", '"'):
        return None
    if re.search(r"\bmodule\b", code) is None or re.search(r"\bendmodule\b", code) is None:
        return None
    if "\\n" in code and "\n" not in code:
        return None
    return code


def process_pyranet(sample, *args, **kwargs):
    try:
        metadata = json.loads(sample['description'])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None
    if not isinstance(metadata, dict):
        return None
    if metadata.get('compile_status') != "No error!":
        return None
    desc = _clean_description(metadata)
    code = _clean_code(sample.get('code'))
    if desc is None or code is None:
        return None
    messages = [
        {'role': 'user', 'content': f"Write the following Verilog program. {desc}"},
        {'role': 'assistant', 'content': code},
    ]
    return data.tokenize_chat(messages, *args, **kwargs)


@data.dataset('bnadimi/PyraNet-Verilog')
def pyranet(split):
    ds = datasets.load_dataset('bnadimi/PyraNet-Verilog', split=split, cache_dir=data.HF_CACHE)
    return ds, process_pyranet
