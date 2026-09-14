"""The optional evaluation bridge serves ordinary chat completions."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
from openai import AsyncOpenAI

import models
import prompting
from models import server


async def request(port):
    client = AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="stx",
        max_retries=0,
    )
    try:
        return await client.chat.completions.create(
            model="stx",
            messages=[{"role": "user", "content": "Count to three."}],
            max_tokens=4,
        )
    finally:
        await client.close()


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
    assert prompting.resolve_prompt_format("auto", model.source) == "text"
    try:
        prompting.validate_prompt_contract(
            model.tokenizer, "chat",
            prompting.DEFAULT_CHAT_TEMPLATE_KWARGS)
    except ValueError as error:
        assert "chat template does not emit tokenizer.eos_token_id" in str(error)
    else:
        raise AssertionError("Qwen Base tokenizer unexpectedly passed chat contract")
    for seed in (1, 2):
        port = server.start(model, seq_len=96, seed=seed)
        assert server._prompt_format == "text"
        response = asyncio.run(request(port))
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens <= 4
        assert response.choices[0].finish_reason in ("stop", "length")
        server.stop()
    print("ok: model-aware prompt format, fixed-batch bridge, and clean restart")


if __name__ == "__main__":
    main()
