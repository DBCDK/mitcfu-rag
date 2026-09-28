#!/usr/bin/env python3
"""Smoke test for the glyph-gate backend swap (phases 1-4), without needing a
FAISS index, embedding model, or Docker -- exercises exactly the code that
changed: gateway auth, dynamic model discovery (GET /v1/models), and a real
streaming chat completion through AgentStreamingGenerator.

It does NOT exercise the LangGraph routing layer or endpoints.py's error
mapping (those need the full service / FAISS index, or are already covered
by the mocked unit tests in tests/test_service.py).

Usage:
    export MITCFU_LLM_GATEWAY_URL="https://llm.dbc.dk"   # or your staging host
    export MITCFU_LLM_GATEWAY_TOKEN="..."
    uv run python myprompts/smoke_test_gateway.py
"""

import asyncio
import logging

from mitcfu_rag.rag.generators.agent_streaming_generator import AgentStreamingGenerator

logging.basicConfig(level=logging.DEBUG)


async def main():
    generator = AgentStreamingGenerator()
    print(f"Discovered models: {generator.backend.available_models}")
    print(f"Default model: {generator.backend.default_model}")

    prompt_template = {"name": "SIMPLE", "prompt": "Answer briefly, in Danish."}
    input_ = {
        "input": [{"role": "user", "content": "Count slowly to 5."}],
        "model": generator.backend.default_model,
    }

    print("\nStreaming response:")
    async for chunk in generator.generate(references=[], input=input_, prompt_template=prompt_template):
        print(chunk, end="", flush=True)
    print()

    # Not calling generator.aclose() here: it's unused in the real service too
    # (nothing calls it in start.py), and closing the client immediately after
    # a single streamed response tends to race the connection's own cleanup,
    # producing a noisy-but-harmless "generator didn't stop after athrow()"
    # warning during asyncio.run()'s shutdown_asyncgens() pass.


if __name__ == "__main__":
    asyncio.run(main())
