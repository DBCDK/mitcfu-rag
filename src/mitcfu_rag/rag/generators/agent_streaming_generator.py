#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.solr_generator - solr_generator

============
EmbeddingGenerator
============

EmbeddingGenerator generates an answer based on a list of references and the query.

example of usage:

    d_generator = EmbeddingGenerator()
    query = "Er der noget om biblioteker?"
    references = ['På visse biblioteker kan du låne fiskestænger, så du kan fange din egen middag efter at have læst om det.']
    response = d_generator(references, query)
    print(f'response: {response}')
"""

import random
import logging
import json
import datetime
import os
import asyncio
import aiohttp
import requests
from mitcfu_rag.rag.rag import Generator, SourcesWithScore, AnswerWithSource, Reference

roles_to_ignore = ["resetter", "summarizer"]

logger = logging.getLogger(__name__)

START_TURN_USER = "<start_of_turn>user\n"
START_TURN_MODEL = "<start_of_turn>model\n"
END_TURN = "<end_of_turn>\n"


class AgentStreamingGenerator(Generator):
    def __init__(self):
        self.streaming_delays = [0.01, 0.02, 0.03]
        #self.chat_bib_url = os.environ.get(
        #    "CHAT_BIB_URL",
        #    "http://chat-bib-tgi-1-0.mi-prod.svc.cloud.dbc.dk/generate_stream",
        #)
        self.mitcfu_tgi_url = os.environ.get("MITCFU_TGI_URL", "http://gemma-3-12b-it.mi-prod.svc.cloud.dbc.dk/v1/chat/completions")
        self.system_message = "Du er MitCFU-Chat. Du hjælper med søgninger i MitCFU kataloget. Du svarer altid på dansk."
        self.prompt_template = None
        self.agent_type = None
        self.missing_reference_prompt = """
Brugeren har stillet et spørgsmål du ikke kan finde nogen kilder om.
Forklar brugeren at du ikke kan finde svaret på spørgsmålet, og bed dem om at omformulere det.
Afslut ALTID dit svar med følgende:
Du kan få hjælp og vejdledning til brug af MitCFU her https://wiki.mitcfu.dk/.
"""
        self.session = aiohttp.ClientSession()

    async def generate(
        self,
        references: list[Reference],
        input: list[dict],
        prompt_template: str = None,
    ):
        logger.info(f"parsed_references: {references}")
        self.prompt_template = prompt_template["prompt"]
        self.agent_type = input["agent"]
        # remove sources from output if generated
        messages = input["input"]
        cleaned_messages = []
        for i, message in enumerate(messages):
            # skip initial welcome message
            if message["role"] == "assistant":
                logger.debug("SPLIT MESSAGES")
                for m in message["content"].split("**Kilder**:"):
                    logger.debug(m)
                logger.debug("END SPLIT MESSAGES")
                message["content"] = message["content"].split("**Kilder**:")[0]
                cleaned_messages.append(message)
            else:
                cleaned_messages.append(message)

        max_new_tokens = 1000 if not self.agent_type == "ROUTER" else 250
        async for chunk in self.llm_generate(
            {
                "messages": cleaned_messages,
                "model": "tgi", "stream": True, "max_tokens": max_new_tokens
            },
            references,
        ):
            yield chunk

    async def async_llm_format(self, msgs, parsed_references):
        await asyncio.sleep(0)
        return self.llm_format(msgs, parsed_references)

    def llm_format(self, msgs, parsed_references):
        # Set start token and add system prompt
        result = START_TURN_USER
        result += f"{self.system_message}"

        # format input for agents that need documents as context
        if self.agent_type in {"RAG", "FOLLOW_UP"}:
            # Only generate something of there are references.
            if parsed_references:
                # Add prompt template, set through input
                result += self.prompt_template
                # Format references
                result += "Dokumenter:" + (
                    ". ".join(
                        [
                            f"{ref.article_headline}: {ref.text[:500]}"
                            for ref in parsed_references
                        ]
                    )
                    + ""
                )
                # End "system" instructions.
                result += END_TURN
                # Format chat history
                for msg in msgs:
                    if msg["role"] == "assistant" or msg["role"] == "user":
                        result += START_TURN_USER if msg["role"] == "user" else START_TURN_MODEL
                        result += f"\n{msg['content']}"
                        result += END_TURN
            else:
                # If no references, use missing reference prompt
                result += self.missing_reference_prompt
                result += END_TURN
        # format agents that need the chathistory as context
        elif self.agent_type in {"REFORMULATOR", "ROUTER"}:
            result += self.prompt_template
            result += "Chat-historik:\n\n"
            for msg in msgs:
                if msg["role"] == "assistant" or msg["role"] == "user":
                    result += "Bruger: " if msg["role"] == "user" else "Model: "
                    result += f"\n{msg['content']}"
            result += END_TURN
        else:
            result += self.prompt_template
            result += END_TURN
            for msg in msgs:
                if msg["role"] == "assistant" or msg["role"] == "user":
                    result += START_TURN_USER if msg["role"] == "user" else START_TURN_MODEL
                    result += f"\n{msg['content']}"
                    result += END_TURN
        # Finally, add model start token at end of prompt
        result += START_TURN_MODEL
        logger.info(f"Input for agent {self.agent_type}:{str(result)}")
        return [{"role": "user", "content": result}]

    def decode(self, input, stream=False):
        try:
            return input.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.debug(f"UnicodeDecodeError: {e}")
            return input.decode("utf-8", errors="ignore")

    async def reference_generator(self, references: list[Reference]):
        for ref in references:
            yield json.dumps({"choices": [{"delta": {"content": "\n"}}]})
            yield json.dumps({"choices": [{"delta": {"content": "\n"}}]})
            tokens = [f"[{ref.article_headline}]({ref.article_link})"]
            #tokens = [f"[{ref.id}](https://mitcfu.dk/MaterialeInfo/?faust={ref.id})"]
            for token in tokens:
                yield json.dumps({"choices": [{"delta": {"content": token}}]})

    async def async_reference_generator(self, parsed_references: list):
        yield json.dumps({"choices": [{"delta": {"content": "\n"}}]})
        await asyncio.sleep(0.01)
        yield json.dumps({"choices": [{"delta": {"content": "\n"}}]})
        await asyncio.sleep(0.01)
        yield json.dumps({"choices": [{"delta": {"content": "**Kilder**"}}]})
        await asyncio.sleep(0.01)
        yield json.dumps({"choices": [{"delta": {"content": ":"}}]})
        await asyncio.sleep(0.01)
        yield json.dumps({"choices": [{"delta": {"content": "\n"}}]})
        await asyncio.sleep(0.01)
        yield json.dumps({"choices": [{"delta": {"content": "\n"}}]})
        async for ref in self.reference_generator(parsed_references):
            await asyncio.sleep(random.choice(self.streaming_delays))
            yield ref
        yield json.dumps({"choices": [{"delta": {"content": END_TURN}}]})

    async def llm_generate(self, input, parsed_references):
        fetch_options = {
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "no-store",
            },
            "method": "POST",
            "redirect": "manual",
        }

        inputs = await asyncio.gather(
            self.async_llm_format(input["messages"], parsed_references)
        )
        request_body = {
            "messages": inputs[0],
            "model": input["model"],
            "stream": input["stream"],
            "max_tokens": input["max_tokens"]
        }
        request_body_str = json.dumps(request_body)
        # if (
        #     not self.agent_type == "RAG"
        # ):  # RAG disabled until we know what CFU wants with the documents
        async with self.session.post(
            self.mitcfu_tgi_url,
            headers=fetch_options["headers"],
            data=request_body_str,
        ) as response:
            async for chunk in response.content.iter_chunked(1024):
                if chunk:
                    # yield chunk
                    decoded_value = self.decode(chunk, stream=True)
                    try:
                        obj = json.loads(decoded_value.replace("data:", ""))
                        for choice in obj.get("choices", []):
                            if token := choice.get("delta", {}).get("content", ""):
                                if token == END_TURN and parsed_references:
                                    yield ""
                                else:
                                    yield chunk
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        logger.info(f"Error during streaming: {e}")

        # filter references so that no two references have the same article_link
        if parsed_references:
            seen_links = set()
            filtered_references = []
            for ref in parsed_references:
                if ref.article_link not in seen_links:
                   seen_links.add(ref.article_link)
                   filtered_references.append(ref)

            async for ref in self.async_reference_generator(filtered_references):
                yield f"data:{ref}\n"
                #yield b'\n'
