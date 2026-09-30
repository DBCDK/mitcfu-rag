#!/usr/bin/env python3

"""
:mod:`mitcfu_rag.service.endpoints` -- FastAPI routes for the streaming
endpoint

No `dbc_pyutils` imports here: DBC integration lives entirely behind
`_dbc_optional`, wired up in `start.create_app`. Everything mounted here
runs identically with or without `dbc_pyutils` installed.
"""

import json
import logging
import resource
import time
import uuid

from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from openai import APIError
from openai import APIStatusError

logger = logging.getLogger(__name__)

router = APIRouter()

# Upstream (glyph-gate) statuses that mean something more specific than "the
# LLM backend broke": an expired/invalid token, a token denied by policy or
# missing the required capability, or a rate limit -- all normal operating
# conditions a caller can act on, not server errors. Anything else upstream
# (including a 5xx from the backend) still maps to a generic 502.
_UPSTREAM_ERROR_TYPES = {
    401: "authentication_error",
    403: "permission_denied",
    429: "rate_limit_exceeded",
}


def _upstream_error_payload(exc: APIError) -> tuple[dict, int]:
    """Map an openai APIError from the glyph-gate backend to an OpenAI-style
    error body and HTTP status code."""
    status_code = exc.status_code if isinstance(exc, APIStatusError) else None
    error_type = _UPSTREAM_ERROR_TYPES.get(status_code, "upstream_error")
    return {"error": {"message": str(exc), "type": error_type}}, status_code if error_type != "upstream_error" else 502


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-chat-style endpoint driving the agentic RAG graph.

    Supports both `stream: true` (SSE `data: ...\\n\\n` chunks terminated by
    `data: [DONE]\\n\\n`) and non-streaming (`chat.completion` JSON object).
    """
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    default_model = request.app.state.default_model
    available_models = request.app.state.available_models
    model_name = body.get("model", default_model)

    if model_name not in available_models:
        return JSONResponse(
            {
                "error": {
                    "message": f"model '{model_name}' not found. Available models: {', '.join(available_models)}",
                    "type": "invalid_request_error",
                    "param": "model",
                }
            },
            status_code=400,
        )

    # The rag pipeline expects content to be a str, not a list of dicts
    messages = [
        {"role": msg["role"], "content": content["text"]}
        for msg in messages
        for content in (
            msg["content"] if isinstance(msg["content"], list) else [{"type": "text", "text": msg["content"]}]
        )
        if content.get("type", "") == "text"
    ]

    agentic_graph = request.app.state.agentic_graph
    try:
        result = await agentic_graph.graph.ainvoke({"input": messages, "model": model_name})
    except APIError as e:
        # Routing (the graph's entry node) always makes an LLM call before any
        # response has started, so an upstream auth/rate-limit/backend error here
        # can still get a proper status code and JSON body -- unlike the mid-stream
        # case below, where the SSE response has already started with a 200.
        logger.warning(f"Upstream LLM error during routing: {e}")
        payload, status_code = _upstream_error_payload(e)
        return JSONResponse(payload, status_code=status_code)

    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def frame(delta=None, finish_reason=None):
        return {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": delta} if delta is not None else {},
                    "finish_reason": finish_reason,
                }
            ],
        }

    if stream:

        async def chunk_generator():
            try:
                async for delta in result["output"]:
                    yield f"data: {json.dumps(frame(delta=delta))}\n\n"
                yield f"data: {json.dumps(frame(finish_reason='stop'))}\n\n"
            except APIError as e:
                logger.warning(f"Upstream LLM error mid-stream: {e}")
                # The stream already started with a 200; the status code half of
                # the mapping can't apply here, only the error body shape.
                error_payload, _ = _upstream_error_payload(e)
                yield f"data: {json.dumps(error_payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(chunk_generator(), media_type="text/event-stream")

    try:
        output = "".join([token async for token in result["output"]])
    except APIError as e:
        logger.warning(f"Upstream LLM error: {e}")
        payload, status_code = _upstream_error_payload(e)
        return JSONResponse(payload, status_code=status_code)
    return JSONResponse(
        {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output},
                    "finish_reason": "stop",
                }
            ],
        }
    )


@router.get("/v1/models")
async def list_models(request: Request):
    """OpenAI-style model listing, so callers (including the Streamlit demo
    UI) can discover which models this deployment can actually route to,
    instead of guessing a `model` value for `/v1/chat/completions`."""
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": model_name, "object": "model", "created": created, "owned_by": "mitcfu-rag"}
            for model_name in request.app.state.available_models
        ],
    }


@router.get("/status")
async def status(request: Request):
    """Service status; enriched with build/instance/stat info when
    `dbc_pyutils` is installed, a bare `{"status": "ok"}` otherwise."""
    state = request.app.state
    if not state.dbc_available:
        return {"status": "ok"}

    info = state.build_info
    return {
        "ok": True,
        "build": info["build_number"],
        "git": info.get("git_revision"),
        "version": info["version"],
        "instance-id": state.instance_id,
        "ab-id": state.ab_id,
        "mem-usage": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "statistics": [stat.describe() for stat in state.stats.values()],
    }
