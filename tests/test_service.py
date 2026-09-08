#!/usr/bin/env python
"""Tests for `mitcfu_rag.service` -- the FastAPI streaming service.

Uses a lightweight fake in place of `AgenticGraph`/`AgenticRAG` so these
tests never load real embedding/torch models. `dbc_pyutils` is only
installed via the non-default `dbc` dependency group, so a plain dev
`.venv` does *not* have it -- the "available" branch is exercised by
monkeypatching `mitcfu_rag.service._dbc_optional` with fakes rather than
depending on the real package being installed.
"""

import json
import types
import unittest
from unittest import mock

from fastapi import Request
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from mitcfu_rag.service import _dbc_optional
from mitcfu_rag.service.start import create_app
from mitcfu_rag.service.start import parse_args


class _FakeGraph:
    """Stand-in for `AgenticGraph.graph`: replays canned content deltas,
    exactly as `AgentStreamingGenerator.generate` yields plain strings
    (no SSE framing -- that's `endpoints.chat_completions`'s job)."""

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


class _FakeStatistics:
    def __init__(self, name=None):
        self.name = name

    def describe(self):
        return {"name": self.name, "total-success": 0, "total-failure": 0}


class _FakePrometheusMiddleware:
    def __init__(self, app, excluded_paths=frozenset()):
        self.app = app
        self.excluded_paths = excluded_paths

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)


async def _fake_metrics_endpoint(request: Request):
    return PlainTextResponse("# fake metrics\n")


def _dbc_available_patches():
    """Fakes standing in for the real `dbc_pyutils` symbols, so the
    "available" gate branch is exercised without needing `dbc_pyutils`
    installed (it only ships via the non-default `dbc` dependency group)."""
    fake_build_info = types.SimpleNamespace(
        get_info=lambda package: {"build_number": "42", "git_revision": "deadbeef", "version": "1.2.3"}
    )
    return mock.patch.multiple(
        _dbc_optional,
        DBC_AVAILABLE=True,
        build_info=fake_build_info,
        create_instance_id=lambda num_digits=8: "test-instance",
        Statistics=_FakeStatistics,
        install_base_handler=lambda app: None,
        PrometheusMiddleware=_FakePrometheusMiddleware,
        metrics_endpoint=_fake_metrics_endpoint,
        setup_logging=lambda: None,
    )


def _build_app(chunks=("hi",)):
    args = parse_args(["embedding-model-path", "faiss-path"])
    with (
        mock.patch("mitcfu_rag.service.start.AgenticRAG", return_value=object()),
        mock.patch("mitcfu_rag.service.start.AgenticGraph", return_value=_FakeAgenticGraph(list(chunks))),
    ):
        return create_app(args)


class TestStatusAndMetricsGating(unittest.TestCase):
    def test_status_enriched_when_dbc_available(self):
        with _dbc_available_patches():
            app = _build_app()
            response = TestClient(app).get("/status")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["instance-id"], "test-instance")
        self.assertEqual(body["statistics"], [{"name": "query", "total-success": 0, "total-failure": 0}])
        self.assertEqual(body["ab-id"], 1)

    def test_metrics_mounted_when_dbc_available(self):
        with _dbc_available_patches():
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
        app = _build_app(chunks=["The", " answer"])
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["choices"][0]["message"], {"role": "assistant", "content": "The answer"})

    def test_streaming_response_frames_deltas_as_sse_and_terminates(self):
        app = _build_app(chunks=["The", " answer"])
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))
        frames = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.split("\n\n")
            if line and line != "data: [DONE]"
        ]
        deltas = [frame["choices"][0]["delta"].get("content") for frame in frames]
        self.assertEqual(deltas, ["The", " answer", None])
        self.assertEqual(frames[-1]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(all(frame["object"] == "chat.completion.chunk" for frame in frames))


if __name__ == "__main__":
    unittest.main()
