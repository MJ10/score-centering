"""ARC-Challenge chat-MCQ benchmark (allenai/ai2_arc)."""
import datasets

from tasks.prefill import chat_mcq


@chat_mcq.benchmark("arc_challenge_chat", seq_len=256)
def _load_arc_challenge_chat(seed=42):
    ds = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = chat_mcq.shuffle_rows(ds, seed)
    return [
        chat_mcq.labeled_choice_example(row["question"], row["choices"], row["answerKey"])
        for row in rows
    ]
