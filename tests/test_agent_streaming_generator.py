#!/usr/bin/env python
"""Tests for `served_model_names` -- fetching the model list from glyph-gate's
GET /v1/models instead of a deploy-time env var list.
"""

import os
import unittest
from unittest import mock

import httpx

from mitcfu_rag.rag.generators import agent_streaming_generator as gen


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, json_error=None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self._json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://gateway.example/v1/models")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._json_data


class _FakeClient:
    """Stand-in for `httpx.Client` as a context manager."""

    def __init__(self, response=None, get_error=None):
        self._response = response
        self._get_error = get_error
        self.requested_url = None
        self.requested_headers = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, headers=None):
        self.requested_url = url
        self.requested_headers = headers
        if self._get_error is not None:
            raise self._get_error
        return self._response


class TestServedModelNames(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(
            os.environ,
            {"MITCFU_LLM_GATEWAY_URL": "https://gateway.example", "MITCFU_LLM_GATEWAY_TOKEN": "test-token"},
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        # Don't let a developer's shell setting reorder the lists below.
        os.environ.pop("MITCFU_DEFAULT_MODEL", None)

    def test_filters_to_chat_completions_capable_models(self):
        response = _FakeResponse(
            json_data={
                "data": [
                    {"id": "model-a", "dbc_model_card": {"capabilities": ["chat_completions"]}},
                    {"id": "model-b", "dbc_model_card": {"capabilities": ["embeddings"]}},
                    {"id": "model-c", "dbc_model_card": {"capabilities": ["chat_completions", "tools"]}},
                ]
            }
        )
        with mock.patch.object(gen.httpx, "Client", return_value=_FakeClient(response)):
            models = gen.served_model_names()
        self.assertEqual(models, ["model-a", "model-c"])

    def test_sends_bearer_token_to_v1_models(self):
        response = _FakeResponse(
            json_data={"data": [{"id": "model-a", "dbc_model_card": {"capabilities": ["chat_completions"]}}]}
        )
        fake_client = _FakeClient(response)
        with mock.patch.object(gen.httpx, "Client", return_value=fake_client):
            gen.served_model_names()
        self.assertEqual(fake_client.requested_url, "https://gateway.example/v1/models")
        self.assertEqual(fake_client.requested_headers, {"Authorization": "Bearer test-token"})

    def test_raises_when_no_models_have_chat_completions_capability(self):
        response = _FakeResponse(
            json_data={"data": [{"id": "model-b", "dbc_model_card": {"capabilities": ["embeddings"]}}]}
        )
        with mock.patch.object(gen.httpx, "Client", return_value=_FakeClient(response)):
            with self.assertRaises(RuntimeError):
                gen.served_model_names()

    def test_raises_when_gateway_unreachable(self):
        fake_client = _FakeClient(get_error=httpx.ConnectError("boom"))
        with mock.patch.object(gen.httpx, "Client", return_value=fake_client):
            with self.assertRaises(RuntimeError):
                gen.served_model_names()

    def test_raises_on_http_error_status(self):
        response = _FakeResponse(status_code=401)
        with mock.patch.object(gen.httpx, "Client", return_value=_FakeClient(response)):
            with self.assertRaises(RuntimeError):
                gen.served_model_names()

    def test_raises_when_gateway_url_env_missing(self):
        del os.environ["MITCFU_LLM_GATEWAY_URL"]
        with self.assertRaises(RuntimeError):
            gen.served_model_names()

    def test_raises_when_gateway_token_env_missing(self):
        del os.environ["MITCFU_LLM_GATEWAY_TOKEN"]
        with self.assertRaises(RuntimeError):
            gen.served_model_names()

    def test_raises_runtimeerror_on_non_json_response(self):
        response = _FakeResponse(json_error=ValueError("not json"))
        with mock.patch.object(gen.httpx, "Client", return_value=_FakeClient(response)):
            with self.assertRaisesRegex(RuntimeError, "Malformed response"):
                gen.served_model_names()

    def test_raises_runtimeerror_on_entry_missing_id(self):
        response = _FakeResponse(
            json_data={"data": [{"dbc_model_card": {"capabilities": ["chat_completions"]}}]}
        )
        with mock.patch.object(gen.httpx, "Client", return_value=_FakeClient(response)):
            with self.assertRaisesRegex(RuntimeError, "Malformed response"):
                gen.served_model_names()

    def _three_model_response(self):
        return _FakeResponse(
            json_data={
                "data": [
                    {"id": "model-a", "dbc_model_card": {"capabilities": ["chat_completions"]}},
                    {"id": "model-b", "dbc_model_card": {"capabilities": ["chat_completions"]}},
                    {"id": "model-c", "dbc_model_card": {"capabilities": ["chat_completions"]}},
                ]
            }
        )

    def test_default_model_env_moves_model_to_front(self):
        os.environ["MITCFU_DEFAULT_MODEL"] = "model-c"
        with mock.patch.object(gen.httpx, "Client", return_value=_FakeClient(self._three_model_response())):
            models = gen.served_model_names()
        self.assertEqual(models, ["model-c", "model-a", "model-b"])

    def test_empty_default_model_env_keeps_gateway_order(self):
        os.environ["MITCFU_DEFAULT_MODEL"] = ""
        with mock.patch.object(gen.httpx, "Client", return_value=_FakeClient(self._three_model_response())):
            models = gen.served_model_names()
        self.assertEqual(models, ["model-a", "model-b", "model-c"])

    def test_raises_when_default_model_env_not_available(self):
        os.environ["MITCFU_DEFAULT_MODEL"] = "model-x"
        with mock.patch.object(gen.httpx, "Client", return_value=_FakeClient(self._three_model_response())):
            with self.assertRaisesRegex(RuntimeError, "MITCFU_DEFAULT_MODEL"):
                gen.served_model_names()

    def test_raises_runtimeerror_on_non_dict_entry(self):
        response = _FakeResponse(json_data={"data": ["not-a-dict"]})
        with mock.patch.object(gen.httpx, "Client", return_value=_FakeClient(response)):
            with self.assertRaisesRegex(RuntimeError, "Malformed response"):
                gen.served_model_names()


if __name__ == "__main__":
    unittest.main()
