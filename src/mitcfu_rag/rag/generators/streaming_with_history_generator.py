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

class EmbeddingGenerator(Generator):

    def __init__(self):
        self.streaming_delays = [0.01, 0.02, 0.03]
        self.chat_bib_url = os.environ.get("CHAT_BIB_URL", None)
        self.system_message = "Du er FaktaChat, en kritisk chatbot der forholder sig til den viden du får fra brugerens kilder. Du svarer altid på dansk."
        self.prompt_template = """
Du modtager et spørgsmål og nogle kilder. Din opgave er at besvare spørgsmål kun ved at bruge informationen i kilderne.
Det er ikke sikkert at nogen af kilderne er relevante for brugerens forespørgsel.
Du overholder følgende regler:
- Du svarer aldrig på spørgsmål, hvor du ikke kan finde svaret i kilderne.
- Du opfinder aldrig kilder.
- Du svarer altid på dansk.
- Hvis ikke du kan finde svaret, forklarer du at du ikke kan finde svaret, og beder dem omformulere spørgsmålet.
- Dit output er kun dit svar, ikke kilderne på dit svar.

Kilder:\n"""

        self.missing_reference_prompt = """
Brugeren har stillet et spørgsmål du ikke kan finde nogen kilder om.
Forklar brugeren at du ikke kan finde svaret på spørgsmålet, og bed dem om at omformulere det.
Afslut ALTID dit svar med følgende:
Du kan tjekke Faktalinks oversigt over temaer (https://faktalink.dk/tema) eller oversigten over emner (https://faktalink.dk/emne) for inspiration.

Dit svar:
"""
        logger.setLevel(logging.DEBUG)

    async def generate(self, references: list[Reference], messages: list[dict], version: str = None):       
        self.version = version
        logger.info(f"parsed_references: {references}")
        
        #remove sources from output if generated
        cleaned_messages = []
        logger.debug("RAW MESSAGES")
        logger.debug(messages)
        logger.debug("END RAW MESSAGES")
        i = 0
        for i, message in enumerate(messages):
            #skip initial welcome message
            if i == 0:
                continue
            elif message["role"] == "assistant":
                logger.debug("SPLIT MESSAGES")
                for m in message["content"].split("Kilder:"):
                    logger.debug(m)
                logger.debug("END SPLIT MESSAGES")
                message["content"] = message["content"].split("Kilder:")[0]
                cleaned_messages.append(message)
            else:
                cleaned_messages.append(message)

        logger.debug("CLEANED MESSAGES")
        logger.debug(cleaned_messages)
        logger.debug("END CLEANED MESSAGES")

        async for chunk in self.llm_generate({"messages": cleaned_messages, "parameters": {"temperature": 0.1, "max_new_tokens": 1200}}, references):
            yield chunk

    async def async_llm_format(self, msgs, parsed_references):
        await asyncio.sleep(0)
        return self.llm_format(msgs, parsed_references)
        

    def llm_format(self, msgs, parsed_references):
        result = "[INST] <<SYS>>\n"
        result += self.system_message
        result += "\n<</SYS>>[/INST]\n\n"
                
        if parsed_references:
            for msg in msgs:
                    if msg["role"] == "assistant" or msg["role"] == "user":
                        result += f"{msg['content']}"
                        result += "[INST]" if msg["role"] == "assistant" else "[/INST]"
            result += self.prompt_template + "\n\n".join([ref.text for ref in parsed_references]) + "\n\n"
            result += "\n[INST] Brugerens spørgsmål:" + msgs[-1]['content'] + "[/INST]Dit svar:"
        else:
            result += self.missing_reference_prompt
        return result
    
    def decode(self, input, stream=False):
        try :
            return input.decode('utf-8')
        except UnicodeDecodeError as e:
            logger.debug(f"UnicodeDecodeError: {e}")
            return input.decode('utf-8', errors='ignore')
        
    def reference_generator(self, references: list[Reference]):
        for ref in references:
            yield json.dumps({"token": {"text": "\n"}})
            yield json.dumps({"token": {"text": "\n"}})
            tokens = [f"[{ref.article_headline}]({ref.article_link})"]
            for token in tokens:
                yield json.dumps({"token": {"text": token}})


    async def async_reference_generator(self, parsed_references: list):
        yield json.dumps({"token": {"text": "\n"}})
        await asyncio.sleep(0.01)
        yield json.dumps({"token": {"text": "\n"}})
        await asyncio.sleep(0.01)
        yield json.dumps({"token": {"text": "Kilder"}})
        await asyncio.sleep(0.01)
        yield json.dumps({"token": {"text": ":"}})
        await asyncio.sleep(0.01)
        yield json.dumps({"token": {"text": "\n"}})
        await asyncio.sleep(0.01)
        yield json.dumps({"token": {"text": "\n"}})
        for ref in self.reference_generator(parsed_references):
            await asyncio.sleep(random.choice(self.streaming_delays))
            yield ref
    
    async def llm_generate(self, input, parsed_references):
        fetch_options = {
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "no-store"
            },
            "method": "POST",
            "redirect": "manual",
        }

        inputs = await asyncio.gather(self.async_llm_format(input["messages"], parsed_references))
        request_body = {
            "inputs": inputs[0],
            "parameters": input["parameters"]
        }
        request_body_str = json.dumps(request_body)
        async with aiohttp.ClientSession() as session:
            async with session.post(self.chat_bib_url, headers=fetch_options["headers"], data=request_body_str) as response:
                async for chunk in response.content.iter_chunked(1024):
                    if chunk:
                        #yield chunk
                        decoded_value = self.decode(chunk, stream=True)
                        try:
                            obj = json.loads(decoded_value.replace("data:", ""))
                            if not obj.get("token", {}).get("text", {}) == "</s>":
                                yield chunk
                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            logger.info(f"Error during streaming: {e}")

        # filter references so that no two references have the same article_link
        seen_links = set()
        filtered_references = []
        for ref in parsed_references:
            if ref.article_link not in seen_links:
                seen_links.add(ref.article_link)
                filtered_references.append(ref)

        async for ref in self.async_reference_generator(filtered_references):
            yield ref


        
