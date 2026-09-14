"""Fixed-batch sampling preserves prompts, replays, and stops at EOS."""
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

import models
from models import sampling


def main():
    checkpoint_dir = "~/.cache/stx/weights"
    model = models.load(
        "Qwen/Qwen3-0.6B-Base",
        checkpoint_dir,
        init="random",
        hidden_size=128,
        num_hidden_layers=2,
        key=jax.random.key(0),
    )
    sample_tokens = jnp.zeros((8, 2), jnp.int32)
    assert jax.eval_shape(
        partial(model.forward, dtype=jnp.bfloat16),
        sample_tokens,
        model.weights,
    ).dtype == jnp.bfloat16
    assert jax.eval_shape(
        partial(model.forward, dtype=jnp.float32),
        sample_tokens,
        model.weights,
    ).dtype == jnp.float32
    assert model.init_kv(8, 2, dtype=jnp.float32)[0].dtype == jnp.float32

    rng = np.random.default_rng(0)
    prompt_lengths = np.asarray([3, 7, 12, 5, 9, 4, 15, 6])
    tokens = np.full(
        (len(prompt_lengths), 48),
        model.tokenizer.pad_token_id,
        np.int32,
    )
    for row, length in enumerate(prompt_lengths):
        tokens[row, :length] = rng.integers(10, 5000, length)
    if model.tokenizer.eos_token_id is not None:
        tokens[0, 1] = model.tokenizer.eos_token_id

    first, token_logprobs, vocab_logprobs, _, _ = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1,
        logprobs_dtype=jnp.float32)
    replay, replay_token_logprobs, replay_vocab_logprobs, _, _ = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1,
        logprobs_dtype=jnp.float32)
    first, replay = np.asarray(first), np.asarray(replay)
    assert np.array_equal(first, replay)
    assert vocab_logprobs.shape == (
        *tokens.shape, model.config["vocab_size"])
    assert vocab_logprobs.dtype == jnp.float32
    recorded = np.take_along_axis(
        np.asarray(vocab_logprobs[:, :-1]),
        first[:, 1:, None],
        axis=-1,
    )[..., 0]
    assert np.array_equal(recorded, np.asarray(token_logprobs[:, :-1]))
    assert np.array_equal(
        np.asarray(token_logprobs),
        np.asarray(replay_token_logprobs),
    )
    assert np.array_equal(
        np.asarray(vocab_logprobs[:, :1]),
        np.asarray(replay_vocab_logprobs[:, :1]),
    )
    for row, length in enumerate(prompt_lengths):
        assert np.array_equal(first[row, :length], tokens[row, :length])

    # In-loop head collection must ship exact values at its ids and preserve
    # exact top-k mass through k=1024. Deterministic replay makes the full and
    # transported distributions directly comparable. Single device: the
    # whole vocabulary is one local block, the highest-collision deployment.
    _, _, _, head_logprobs, head_ids = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1,
        logprobs_topk=1024)
    full = np.asarray(vocab_logprobs)
    live = np.asarray(token_logprobs) != 0.0
    gathered = np.take_along_axis(full, np.asarray(head_ids), axis=-1)
    assert np.allclose(gathered[live], np.asarray(head_logprobs)[live])
    for k in (128, 1024):
        exact = np.partition(full, -k, axis=-1)[..., -k:]
        exact_mass = np.exp(exact).sum(-1)
        transported_mass = np.exp(np.asarray(head_logprobs)[..., :k]).sum(-1)
        assert np.allclose(
            transported_mass[live], exact_mass[live], rtol=1e-6, atol=1e-7
        ), f"k={k} transported head measurably loses probability mass"

    prompt_eos, _, _, _, _ = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths)
    prompt_eos = np.asarray(prompt_eos)
    assert np.any(
        prompt_eos[0, prompt_lengths[0]:]
        != model.tokenizer.pad_token_id
    ), "an EOS inside the teacher-forced prompt must not stop generation"

    # KV fake-quant: teacher-force one fixed sequence and score it under
    # increasingly coarse cache writes — mismatch must be monotone in bits.
    forced = np.full(len(prompt_lengths), tokens.shape[1])
    scores = {}
    for bits in (0, 8, 3):
        _, forced_logprobs, _, _, _ = sampling.generate(
            jax.random.key(1), model, first, forced, eos_id=-1,
            forward=partial(model.forward, dtype=jnp.bfloat16,
                            kv_quant_bits=bits))
        scores[bits] = np.asarray(forced_logprobs)
    absdiff = {b: np.abs(scores[b] - scores[0]).mean() for b in (8, 3)}
    assert 0 < absdiff[8] < absdiff[3]

    eos_id = int(first[0, prompt_lengths[0] + 2])
    stopped, _, _, _, _ = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=eos_id)
    stopped = np.asarray(stopped)
    eos_position = next(
        position
        for position in range(prompt_lengths[0], stopped.shape[1])
        if stopped[0, position] == eos_id
    )
    assert np.all(
        stopped[0, eos_position + 1:] == model.tokenizer.pad_token_id)
    print("ok: one fixed batch, exact logprobs and top-k, deterministic "
          "replay, KV fake-quant, EOS stop")


if __name__ == "__main__":
    main()
