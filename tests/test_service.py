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

import httpx
from fastapi import Request
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient
from openai import APIConnectionError
from openai import APIStatusError

from mitcfu_rag.service import _dbc_optional
from mitcfu_rag.service.start import create_app
from mitcfu_rag.service.start import parse_args


FAKE_MODELS = ["gemma-4-26b-a4b-it", "other-model"]


def _status_error(status_code, message="upstream denied"):
    """Build an `openai.APIStatusError` as if raised by a real HTTP response
    from the gateway, for exercising `endpoints._upstream_error_payload`."""
    request = httpx.Request("POST", "http://gateway.example/v1/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": {"message": message}})
    return APIStatusError(message, response=response, body={"error": {"message": message}})


def _connection_error(message="connection failed"):
    """Build an `openai.APIConnectionError` (no status code) -- the generic,
    non-4xx-specific upstream failure case."""
    request = httpx.Request("POST", "http://gateway.example/v1/chat/completions")
    return APIConnectionError(message=message, request=request)


class _FakeGraph:
    """Stand-in for `AgenticGraph.graph`: replays canned content deltas,
    exactly as `AgentStreamingGenerator.generate` yields plain strings
    (no SSE framing -- that's `endpoints.chat_completions`'s job)."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.received_states = []

    async def ainvoke(self, state):
        self.received_states.append(state)

        async def _output():
            for chunk in self._chunks:
                yield chunk

        return {"output": _output()}


class _FakeAgenticGraph:
    def __init__(self, chunks):
        self.graph = _FakeGraph(chunks)


class _RaisingGraph:
    """Stand-in for `AgenticGraph.graph` whose `ainvoke` raises immediately --
    simulates an upstream error during the router's own LLM call, before any
    response has started."""

    def __init__(self, exc):
        self._exc = exc
        self.received_states = []

    async def ainvoke(self, state):
        self.received_states.append(state)
        raise self._exc


class _RaisingAgenticGraph:
    def __init__(self, exc):
        self.graph = _RaisingGraph(exc)


class _MidStreamErrorGraph:
    """`ainvoke` succeeds (routing worked); the output generator yields some
    chunks and then raises -- simulates an upstream error during the final
    agent's own generation, after the response may have already started."""

    def __init__(self, chunks, exc):
        self._chunks = chunks
        self._exc = exc
        self.received_states = []

    async def ainvoke(self, state):
        self.received_states.append(state)

        async def _output():
            for chunk in self._chunks:
                yield chunk
            raise self._exc

        return {"output": _output()}


class _MidStreamErrorAgenticGraph:
    def __init__(self, chunks, exc):
        self.graph = _MidStreamErrorGraph(chunks, exc)


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


def _build_app(chunks=("hi",), models=FAKE_MODELS, agentic_graph=None):
    args = parse_args(["embedding-model-path", "faiss-path"])
    with (
        mock.patch("mitcfu_rag.service.start.AgenticRAG", return_value=object()),
        mock.patch(
            "mitcfu_rag.service.start.AgenticGraph",
            return_value=agentic_graph or _FakeAgenticGraph(list(chunks)),
        ),
        mock.patch("mitcfu_rag.service.start.served_model_names", return_value=list(models)),
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

    def test_defaults_to_first_configured_model_when_omitted(self):
        app = _build_app()
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )

        state = app.state.agentic_graph.graph.received_states[-1]
        self.assertEqual(state["model"], FAKE_MODELS[0])

    def test_routes_explicit_model_through_to_the_graph(self):
        app = _build_app()
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "other-model",
                "stream": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "other-model")
        state = app.state.agentic_graph.graph.received_states[-1]
        self.assertEqual(state["model"], "other-model")

    def test_rejects_unknown_model(self):
        app = _build_app()
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "no-such-model",
                "stream": False,
            },
        )

        self.assertEqual(response.status_code, 400)
        error = response.json()["error"]
        self.assertIn("no-such-model", error["message"])
        self.assertEqual(error["param"], "model")
        self.assertEqual(app.state.agentic_graph.graph.received_states, [])


class TestChatCompletionsUpstreamErrors(unittest.TestCase):
    """Upstream (glyph-gate) auth/rate-limit errors should surface as specific,
    actionable HTTP statuses rather than a blanket 502 -- see
    `endpoints._upstream_error_payload`."""

    def test_401_during_routing_returns_authentication_error(self):
        app = _build_app(agentic_graph=_RaisingAgenticGraph(_status_error(401, "token invalid")))
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )

        self.assertEqual(response.status_code, 401)
        error = response.json()["error"]
        self.assertEqual(error["type"], "authentication_error")
        self.assertIn("token invalid", error["message"])

    def test_403_during_routing_returns_permission_denied(self):
        app = _build_app(agentic_graph=_RaisingAgenticGraph(_status_error(403, "app_not_allowed_for_model")))
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["type"], "permission_denied")

    def test_429_during_routing_returns_rate_limit_exceeded(self):
        app = _build_app(agentic_graph=_RaisingAgenticGraph(_status_error(429, "rate limited")))
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["error"]["type"], "rate_limit_exceeded")

    def test_generic_upstream_failure_during_routing_returns_502(self):
        app = _build_app(agentic_graph=_RaisingAgenticGraph(_connection_error()))
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["type"], "upstream_error")

    def test_non_streaming_error_during_generation_maps_status(self):
        agentic_graph = _MidStreamErrorAgenticGraph(["partial "], _status_error(403, "denied mid-generation"))
        app = _build_app(agentic_graph=agentic_graph)
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["type"], "permission_denied")

    def test_streaming_error_during_generation_yields_error_frame_then_done(self):
        agentic_graph = _MidStreamErrorAgenticGraph(["partial "], _status_error(403, "denied mid-generation"))
        app = _build_app(agentic_graph=agentic_graph)
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

        # SSE already started with a 200 before the error occurred -- the status
        # code can't change mid-stream, only the frame contents.
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))
        frames = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.split("\n\n")
            if line and line != "data: [DONE]"
        ]
        error_frames = [frame for frame in frames if "error" in frame]
        self.assertEqual(len(error_frames), 1)
        self.assertEqual(error_frames[0]["error"]["type"], "permission_denied")


class TestListModels(unittest.TestCase):
    def test_lists_configured_models(self):
        app = _build_app()
        response = TestClient(app).get("/v1/models")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "list")
        self.assertEqual([m["id"] for m in body["data"]], FAKE_MODELS)
        self.assertTrue(all(m["object"] == "model" for m in body["data"]))


if __name__ == "__main__":
    unittest.main()
