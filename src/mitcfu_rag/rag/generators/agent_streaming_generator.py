#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`mitcfu_rag.rag.generators.agent_streaming_generator` -- agent_streaming_generator model

============
AgentStreamingGenerator
============

AgentStreamingGenerator generates an answer based on a list of references and the query.

example of usage:

    as_generator = AgentStreamingGenerator()
    query = "Er der noget om biblioteker?"
    references = ['På visse biblioteker kan du låne fiskestænger, så du kan fange din egen middag efter at have læst om det.']
    response = as_generator(references, query)
    print(f'response: {response}')
"""

import random
import logging
import os
import asyncio
from dataclasses import dataclass
import httpx
from openai import AsyncOpenAI
from mitcfu_rag.rag.rag import Generator, Reference
from mitcfu_rag.tools.message_history import clean_sources_from_messages

logger = logging.getLogger(__name__)


def _gateway_base_url() -> str:
    """Read MITCFU_LLM_GATEWAY_URL: the glyph-gate host, e.g. https://llm.dbc.dk.

    Callers append the /v1 path themselves (AsyncOpenAI.base_url wants the .../v1
    prefix and appends chat/completions itself; the raw model-list fetch below
    hits /v1/models directly).
    """
    url = os.environ.get("MITCFU_LLM_GATEWAY_URL")
    if not url:
        raise RuntimeError("MITCFU_LLM_GATEWAY_URL must be set to the glyph-gate base URL (e.g. https://llm.dbc.dk)")
    return url.rstrip("/")


def _gateway_token() -> str:
    """Read MITCFU_LLM_GATEWAY_TOKEN: the bearer token issued by llm-access for this app.

    Never log this value.
    """
    token = os.environ.get("MITCFU_LLM_GATEWAY_TOKEN")
    if not token:
        raise RuntimeError("MITCFU_LLM_GATEWAY_TOKEN must be set to a glyph-gate bearer token")
    return token


def _max_tokens() -> int | None:
    """Read MITCFU_MAX_TOKENS. Unset or empty means no limit (None)."""
    value = os.environ.get("MITCFU_MAX_TOKENS")
    return int(value) if value else None


def _llm_timeout_seconds() -> float:
    """Read MITCFU_LLM_TIMEOUT_SECONDS. The openai SDK default (600s read, 2
    retries) can leave an interactive chat request hanging for up to 30
    minutes on a stalled vLLM backend.
    """
    return float(os.environ.get("MITCFU_LLM_TIMEOUT_SECONDS", "60"))


def served_model_names() -> list[str]:
    """Fetch the chat-completions-capable models this app's glyph-gate token can see.

    Calls GET /v1/models on the gateway instead of reading a deploy-time env var list:
    glyph-gate already filters that response to what the token is authorized for. This
    additionally filters to models whose model card advertises the chat_completions
    capability, since /v1/models can also list audio- or embeddings-only models. The
    first entry is the default used when a request doesn't specify `model`.

    Raises RuntimeError if the gateway is unreachable, denies the request, or the
    resulting list is empty - there is no safe default model to fall back to.
    """
    base_url = _gateway_base_url()
    token = _gateway_token()
    try:
        with httpx.Client(timeout=_llm_timeout_seconds()) as client:
            response = client.get(f"{base_url}/v1/models", headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Failed to fetch model list from glyph-gate at {base_url}: {exc}") from exc

    try:
        models = []
        for entry in response.json().get("data", []):
            card = entry.get("dbc_model_card") or {}
            if "chat_completions" in (card.get("capabilities") or []):
                models.append(entry["id"])
    except (ValueError, KeyError, AttributeError) as exc:
        # ValueError covers json.JSONDecodeError (a subclass); KeyError/AttributeError
        # cover a response that's valid JSON but doesn't match the documented shape
        # (missing "id", non-dict entry/model card, etc). Same fail-fast treatment as
        # the network-error case above: a malformed response is just as unusable as
        # an unreachable gateway, and deserves the same clear, greppable message.
        raise RuntimeError(f"Malformed response from glyph-gate at {base_url}: {exc}") from exc

    if not models:
        raise RuntimeError(f"No chat_completions-capable models available to this glyph-gate token at {base_url}")
    return models


@dataclass(frozen=True)
class ModelBackend:
    client: AsyncOpenAI
    available_models: list[str]

    @property
    def default_model(self) -> str:
        return self.available_models[0]

    def resolve(self, requested_model: str | None) -> str:
        """Returns `requested_model` if it's one of this backend's served
        models, or the default (first) model if none was requested.

        The primary check on a bad `model` lives in `endpoints.chat_completions`
        (it fails the request before any retrieval work runs); this is a
        defensive fallback for other callers of the generator.
        """
        if requested_model is None:
            return self.default_model
        if requested_model not in self.available_models:
            raise ValueError(f"Unknown model {requested_model!r}. Available models: {', '.join(self.available_models)}")
        return requested_model


class AgentStreamingGenerator(Generator):
    def __init__(self, available_models: list[str]):
        """`available_models` is fetched once by the caller (`start.create_app`,
        via `served_model_names()`) and passed in rather than fetched again here --
        two independent live calls to glyph-gate's `/v1/models` could otherwise
        return different lists, letting a model pass `endpoints.py`'s validation
        against one list but fail `ModelBackend.resolve()`'s check against the
        other.
        """
        self.streaming_delays = [0.01, 0.02, 0.03]
        self.backend = ModelBackend(
            client=AsyncOpenAI(
                base_url=f"{_gateway_base_url()}/v1",
                api_key=_gateway_token(),
                timeout=httpx.Timeout(_llm_timeout_seconds(), connect=5.0),
            ),
            available_models=available_models,
        )
        self.max_tokens = _max_tokens()
        self.system_message = (
            "Du er MitCFU-Chat. Du hjælper med søgninger i MitCFU kataloget. Du svarer altid på dansk."
        )
        self.missing_reference_prompt = """
Brugeren har stillet et spørgsmål du ikke kan finde nogen kilder om.
Forklar brugeren at du ikke kan finde svaret på spørgsmålet, og bed dem om at omformulere det.
Afslut ALTID dit svar med følgende:
Du kan få hjælp og vejdledning til brug af MitCFU her https://wiki.mitcfu.dk/.
"""

    async def aclose(self):
        await self.backend.client.close()

    async def generate(
        self,
        references: list[Reference],
        input: dict,
        prompt_template: dict,
    ):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"parsed_references: {references}")
        model_name = self.backend.resolve(input.get("model"))
        logger.info(f"Replying as {prompt_template['name']} with model {model_name}")
        messages = input["input"]
        cleaned_messages = clean_sources_from_messages(messages)

        async for chunk in self.llm_generate(
            {
                "messages": cleaned_messages,
                "prompt_template": prompt_template["prompt"],
                "agent_type": prompt_template["name"],
                "model": model_name,
            },
            references,
        ):
            yield chunk

        # filter references so that no two references have the same article_link
        if references and prompt_template["name"] in {"RAG", "FOLLOW_UP"}:
            seen_links = set()
            filtered_references = []
            for ref in references:
                if ref.article_link not in seen_links:
                    seen_links.add(ref.article_link)
                    filtered_references.append(ref)
            async for ref in self.async_reference_generator(filtered_references):
                yield ref

    def build_messages(
        self,
        msgs: list[dict],
        prompt_template: str,
        agent_type: str,
        parsed_references: list[Reference],
    ) -> list[dict]:
        if agent_type in {"RAG", "FOLLOW_UP"}:
            # Only generate something if there are references.
            if parsed_references:
                system = (
                    self.system_message
                    + "\n"
                    + prompt_template
                    + "\nDokumenter:"
                    + ". ".join(f"{ref.article_headline}: {ref.text[:500]}" for ref in parsed_references)
                )
                return [{"role": "system", "content": system}, *msgs]
            # If no references, use missing reference prompt.
            # Preserves today's behavior: no chat history included on this path.
            system = self.system_message + "\n" + self.missing_reference_prompt
            return [{"role": "system", "content": system}]
        # ROUTER / REFORMULATOR / SIMPLE / FALLBACK: chat history as real messages.
        system = self.system_message + "\n" + prompt_template
        return [{"role": "system", "content": system}, *msgs]

    async def llm_generate(self, input: dict, parsed_references: list[Reference]):
        messages = self.build_messages(
            input["messages"],
            input["prompt_template"],
            input["agent_type"],
            parsed_references,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Input for agent {input['agent_type']}: {messages}")
        create_kwargs = {
            "model": input["model"],
            "messages": messages,
            "stream": True,
            "temperature": 0.1,
        }
        if self.max_tokens is not None:
            create_kwargs["max_tokens"] = self.max_tokens
        stream = await self.backend.client.chat.completions.create(**create_kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning", None)
            if reasoning and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Reasoning: {reasoning}")
            if delta.content:
                yield delta.content

    async def reference_generator(self, references: list[Reference]):
        for ref in references:
            yield "\n\n"
            yield f"[{ref.article_headline}]({ref.article_link})"

    async def async_reference_generator(self, parsed_references: list[Reference]):
        yield "\n\n**Kilder**:\n\n"
        async for ref in self.reference_generator(parsed_references):
            await asyncio.sleep(random.choice(self.streaming_delays))
            yield ref
