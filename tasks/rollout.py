"""Fixed-batch rollouts for single-turn verifiers environments."""
import asyncio
import importlib

import numpy as np
import verifiers as vf

import prompting


def load(environment_id, **arguments):
    try:
        if "/" not in environment_id:
            module = importlib.import_module(f"tasks.zoo.{environment_id.replace('-', '_')}")
            return module.load_environment(**arguments)
    except ImportError:
        pass
    return vf.load_environment(environment_id, **arguments)


def prompt_buffer(
        rows, tokenizer, seq_len, *, prompt_format,
        chat_template_kwargs=None):
    tokens = np.full((len(rows), seq_len), tokenizer.pad_token_id, np.int32)
    prompt_lengths = np.zeros(len(rows), np.int32)
    for index, row in enumerate(rows):
        prompt = prompting.tokenize_prompt(
            row["prompt"], tokenizer, prompt_format,
            chat_template_kwargs)[:seq_len]
        tokens[index, :len(prompt)] = prompt
        prompt_lengths[index] = len(prompt)
    return tokens, prompt_lengths


def completions(tokens, prompt_lengths, tokenizer):
    mask = np.zeros(tokens.shape, bool)
    lengths = np.zeros(tokens.shape[0], np.int32)
    texts = []
    stop_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    for index, prompt_length in enumerate(prompt_lengths):
        generated = tokens[index, prompt_length:]
        end = next(
            (offset for offset, token in enumerate(generated)
             if token in stop_ids),
            len(generated),
        )
        length = end + (
            end < len(generated) and generated[end] == tokenizer.eos_token_id)
        mask[index, prompt_length:prompt_length + length] = True
        lengths[index] = length
        texts.append(
            tokenizer.decode(generated[:end], skip_special_tokens=True))
    return mask, lengths, texts


def score(rubric, rows, texts, group_size):
    states = [
        vf.State(
            input=row,
            prompt=row["prompt"],
            trajectory=[],
            completion=[vf.AssistantMessage(content=text)],
            answer=row.get("answer", ""),
            info=row.get("info", {}),
        )
        for row, text in zip(rows, texts)
    ]

    async def score_all():
        if rubric.has_group_rewards:
            for start in range(0, len(states), group_size):
                await rubric.score_group(states[start:start + group_size])
        else:
            await asyncio.gather(
                *(rubric.score_rollout(state) for state in states))
        await asyncio.gather(*(rubric.cleanup(state) for state in states))

    asyncio.run(score_all())
    return np.asarray(
        [state.get("reward") or 0.0 for state in states], np.float32)
