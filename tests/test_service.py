#!/usr/bin/env python
"""Tests for `mitcfu_rag.service` -- the FastAPI streaming service.

Uses a lightweight fake in place of `AgenticGraph`/`AgenticRAG` so these
tests never load real embedding/torch models, and monkeypatches
`mitcfu_rag.service._dbc_optional.DBC_AVAILABLE` to exercise both the
"dbc_pyutils installed" and "not installed" code paths deterministically
(this repo's dev `.venv` has `dbc_pyutils` installed, so the "available"
branch also runs for real, without extra setup).
"""

import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from mitcfu_rag.service import _dbc_optional
from mitcfu_rag.service.start import create_app
from mitcfu_rag.service.start import parse_args


def _sse_chunk(token: str) -> str:
    """An SSE chunk shaped like `AgentStreamingGenerator`'s real output."""
    return f"data: {json.dumps({'choices': [{'delta': {'content': token}}]})}\n\n"


class _FakeGraph:
    """Stand-in for `AgenticGraph.graph`: replays canned SSE chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    async def ainvoke(self, state):
        async def _output():
            for chunk in self._chunks:
                yield chunk

        return {"output": _output()}


class _FakeAgenticGraph:
    def __init__(self, chunks):
        self.graph = _FakeGraph(chunks)


def _build_app(chunks=(_sse_chunk("hi"),)):
    args = parse_args(["embedding-model-path", "faiss-path"])
    with (
        mock.patch("mitcfu_rag.service.start.AgenticRAG", return_value=object()),
        mock.patch("mitcfu_rag.service.start.AgenticGraph", return_value=_FakeAgenticGraph(list(chunks))),
    ):
        return create_app(args)


class TestStatusAndMetricsGating(unittest.TestCase):
    def test_status_enriched_when_dbc_available(self):
        app = _build_app()
        response = TestClient(app).get("/status")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertIn("instance-id", body)
        self.assertIn("statistics", body)
        self.assertEqual(body["ab-id"], 1)

    def test_metrics_mounted_when_dbc_available(self):
        app = _build_app()
        response = TestClient(app).get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["content-type"])

    def test_status_degrades_when_dbc_unavailable(self):
        with mock.patch.object(_dbc_optional, "DBC_AVAILABLE", False):
            app = _build_app()
            response = TestClient(app).get("/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_metrics_absent_when_dbc_unavailable(self):
        with mock.patch.object(_dbc_optional, "DBC_AVAILABLE", False):
            app = _build_app()
            response = TestClient(app).get("/metrics")

        self.assertEqual(response.status_code, 404)


class TestChatCompletions(unittest.TestCase):
    def test_non_streaming_response_shape(self):
        app = _build_app(chunks=[_sse_chunk("The"), _sse_chunk(" answer")])
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["choices"][0]["message"], {"role": "assistant", "content": "The answer"})

    def test_streaming_response_passes_through_sse_chunks_and_terminates(self):
        app = _build_app(chunks=[_sse_chunk("The"), _sse_chunk(" answer")])
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))
        self.assertIn("The", response.text)
        self.assertIn("answer", response.text)


if __name__ == "__main__":
    unittest.main()
