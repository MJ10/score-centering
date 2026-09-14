"""MMLU (cais/mmlu), both prefill styles: base-model abcd logprobs ('mmlu')
and chat-template ANSWER-letter scoring ('mmlu_chat').
"""
import datasets

from tasks.prefill import chat_mcq
from tasks.prefill.mcq import benchmark

MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


@benchmark('mmlu', style='abcd', seq_len=768)
def _load_mmlu():
    # following lm-eval format
    ds = datasets.load_dataset('cais/mmlu', 'all', split='test')
    return [[
        f"{x['question']}\nA. {x['choices'][0]}\nB. {x['choices'][1]}\nC. {x['choices'][2]}\nD. {x['choices'][3]}\nAnswer:",
        [' A', ' B', ' C', ' D'],
        x['answer'],
    ] for x in ds]


@chat_mcq.benchmark("mmlu_chat", seq_len=512, default_size=2048)
def _load_mmlu_chat(seed=42):
    ds = datasets.load_dataset("cais/mmlu", "all", split="test", revision=MMLU_REVISION)
    rows = chat_mcq.shuffle_rows(ds, seed)
    return [
        [
            chat_mcq.format_prompt(row["question"], row["choices"]),
            chat_mcq.answer_completions(len(row["choices"])),
            int(row["answer"]),
        ]
        for row in rows
    ]
