#!/usr/bin/env python3

"""
:mod:`mitcfu_rag.service.endpoints` -- FastAPI routes for the streaming
endpoint

No `dbc_pyutils` imports here: DBC integration lives entirely behind
`_dbc_optional`, wired up in `start.create_app`. Everything mounted here
runs identically with or without `dbc_pyutils` installed.
"""

import resource
import time
import uuid

from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from mitcfu_rag.tools.llm_formatting import async_gen_wrapper

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

    if stream:

        async def chunk_generator():
            async for chunk in result["output"]:
                yield chunk
            yield "data: [DONE]\n\n"

        return StreamingResponse(chunk_generator(), media_type="text/event-stream")

    output = "".join([token async for token in async_gen_wrapper(result["output"], default_model)])
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
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
