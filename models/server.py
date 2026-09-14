"""A small OpenAI-compatible server for external evaluation harnesses."""
import threading
import time

import jax
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import prompting
from . import sampling


app = FastAPI()
_model = _mesh = _key = _http_server = _http_thread = None
_seq_len = _batch_size = 0
_model_id = "stx"
_prompt_format = "chat"
_chat_template_kwargs = {}


class Request(BaseModel):
    model: str = ""
    messages: list[dict]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    chat_template_kwargs: dict = Field(default_factory=dict)


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{
        "id": _model_id, "object": "model", "owned_by": "stx"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global _key
    try:
        prompt = prompting.tokenize_prompt(
            request.messages,
            _model.tokenizer,
            _prompt_format,
            {**_chat_template_kwargs, **request.chat_template_kwargs},
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if len(prompt) >= _seq_len:
        raise HTTPException(
            400, f"prompt has {len(prompt)} tokens; seq_len={_seq_len}")

    tokens = np.full((_batch_size, _seq_len), _model.tokenizer.pad_token_id, np.int32)
    tokens[:, :len(prompt)] = prompt
    _key, sample_key = jax.random.split(_key)
    jax.set_mesh(_mesh)
    tokens, _, _, _, _ = sampling.generate(
        sample_key,
        _model,
        tokens,
        np.full(_batch_size, len(prompt), np.int32),
    )
    tokens = np.asarray(tokens)[0]

    requested = request.max_tokens if request.max_tokens is not None else request.max_completion_tokens
    limit = _seq_len if requested is None else min(
        len(prompt) + max(requested, 0), _seq_len)
    stop_ids = {_model.tokenizer.eos_token_id, _model.tokenizer.pad_token_id}
    end = next(
        (position for position in range(len(prompt), limit)
         if tokens[position] in stop_ids),
        limit,
    )
    stopped = end < limit
    content = _model.tokenizer.decode(tokens[len(prompt):end], skip_special_tokens=True)
    completion_tokens = end - len(prompt)
    return {
        "id": f"chatcmpl-{time.time_ns()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "index": 0,
            "finish_reason": "stop" if stopped else "length",
        }],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": completion_tokens,
            "total_tokens": len(prompt) + completion_tokens,
        },
    }


def start(
        model, *, seq_len, port=0, model_id="stx", seed=0,
        prompt_format="auto", chat_template_kwargs=None):
    global _model, _mesh, _key, _seq_len, _batch_size, _model_id
    global _prompt_format, _chat_template_kwargs
    global _http_server, _http_thread
    if _http_thread is not None:
        raise RuntimeError("server is already running")
    _model = model
    _mesh = jax.tree.leaves(model.weights)[0].sharding.mesh
    _key = jax.random.key(seed)
    _seq_len = seq_len
    # Eight rows also keeps the CPU fallback away from unsupported batch-one
    # bfloat16 kernels; real accelerator meshes use at least one row/device.
    _batch_size = max(8, jax.device_count())
    _model_id = model_id
    _prompt_format = prompting.resolve_prompt_format(
        prompt_format, model.source)
    _chat_template_kwargs = {
        **prompting.DEFAULT_CHAT_TEMPLATE_KWARGS,
        **(chat_template_kwargs or {}),
    }
    prompting.validate_prompt_contract(
        model.tokenizer, _prompt_format, _chat_template_kwargs)
    # Compile on the caller thread; some JAX backends cannot compile from
    # uvicorn's event-loop thread.
    jax.set_mesh(_mesh)
    warmup = np.full((_batch_size, _seq_len), model.tokenizer.pad_token_id, np.int32)
    if model.tokenizer.eos_token_id is not None and _seq_len > 1:
        warmup[:, 1] = model.tokenizer.eos_token_id
    jax.block_until_ready(sampling.generate(
        _key, model, warmup,
        np.full(_batch_size, 2, np.int32)))
    _http_server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning"))
    _http_thread = threading.Thread(target=_http_server.run, daemon=True)
    _http_thread.start()
    while not _http_server.started:
        time.sleep(0.01)
    return _http_server.servers[0].sockets[0].getsockname()[1]


def stop():
    global _http_server, _http_thread
    if _http_server is not None:
        _http_server.should_exit = True
    if _http_thread is not None:
        _http_thread.join()
    _http_server = _http_thread = None
