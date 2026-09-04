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

logger = logging.getLogger(__name__)

router = APIRouter()


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
    model_name = body.get("model", default_model)

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
    result = await agentic_graph.graph.ainvoke({"input": messages})

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
                error_frame = {"error": {"message": str(e), "type": e.__class__.__name__}}
                yield f"data: {json.dumps(error_frame)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(chunk_generator(), media_type="text/event-stream")

    try:
        output = "".join([token async for token in result["output"]])
    except APIError as e:
        logger.warning(f"Upstream LLM error: {e}")
        return JSONResponse(
            {"error": {"message": str(e), "type": e.__class__.__name__}},
            status_code=502,
        )
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
