"""CommonsenseQA chat-MCQ benchmark (tau/commonsense_qa)."""
import datasets

from tasks.prefill import chat_mcq


@chat_mcq.benchmark("commonsenseqa_chat", seq_len=192)
def _load_commonsenseqa_chat(seed=42):
    ds = datasets.load_dataset("tau/commonsense_qa", split="validation")
    rows = chat_mcq.shuffle_rows(ds, seed)
    return [
        chat_mcq.labeled_choice_example(row["question"], row["choices"], row["answerKey"])
        for row in rows
    ]
