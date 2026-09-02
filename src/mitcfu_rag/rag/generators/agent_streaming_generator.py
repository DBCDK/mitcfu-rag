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
from openai import AsyncOpenAI
from mitcfu_rag.rag.rag import Generator, Reference
from mitcfu_rag.tools.llm_formatting import clean_sources_from_messages
from mitcfu_rag.config import GEMMA_4_26B

logger = logging.getLogger(__name__)


def _vllm_base_url() -> str:
    """Read MITCFU_VLLM_URL and strip a trailing /chat/completions if present.

    Today's deployed env value is the full completions URL; AsyncOpenAI.base_url
    wants the .../v1 prefix and appends chat/completions itself. This strip is a
    defensive safety net, not a long-term crutch.
    """
    url = os.environ.get(
        "MITCFU_VLLM_URL",
        "http://vllm-gemma-4-26b-a4b-1-0.ai-staging.svc.cloud.dbc.dk/v1/chat/completions",
    )
    return url.removesuffix("/chat/completions")


class AgentStreamingGenerator(Generator):
    def __init__(self):
        self.streaming_delays = [0.01, 0.02, 0.03]
        self.request_model_names = {
            GEMMA_4_26B: os.environ.get("MITCFU_VLLM_MODEL", ""),
        }
        self.clients: dict[str, AsyncOpenAI] = {
            GEMMA_4_26B: AsyncOpenAI(base_url=_vllm_base_url(), api_key="unused"),
        }
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
        for client in self.clients.values():
            await client.close()

    async def generate(
        self,
        references: list[Reference],
        input: list[dict],
        prompt_template: str = None,
    ):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"parsed_references: {references}")
        logger.info(f"Replying as {prompt_template['name']} with model {prompt_template['model']}")
        messages = input["input"]
        cleaned_messages = clean_sources_from_messages(messages)

        async for chunk in self.llm_generate(
            {
                "messages": cleaned_messages,
                "model_name": prompt_template["model"],
                "prompt_template": prompt_template["prompt"],
                "agent_type": prompt_template["name"],
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
        model_key = input["model_name"]
        client = self.clients[model_key]
        messages = self.build_messages(
            input["messages"],
            input["prompt_template"],
            input["agent_type"],
            parsed_references,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Input for agent {input['agent_type']}: {messages}")
        stream = await client.chat.completions.create(
            model=self.request_model_names.get(model_key, model_key),
            messages=messages,
            stream=True,
            max_tokens=1000,
            temperature=0.1,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
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
