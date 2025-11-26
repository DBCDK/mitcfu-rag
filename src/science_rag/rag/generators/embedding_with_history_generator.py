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
import requests
from huggingface_hub import InferenceClient
from mitcfu_rag.rag.rag import Generator, SourcesWithScore, AnswerWithSource

roles_to_ignore = ["resetter", "summarizer"]

logger = logging.getLogger(__name__)


class EmbeddingGenerator(Generator):
    def __init__(self):
        self.system_message = "Du er FaktaChat, en kritisk chatbot der forholder sig til den viden du får fra brugerens kilder. Du svarer altid på dansk."
        self.prompt_template = """
Du modtager et spørgsmål og nogle kilde. Din opgave er at besvare spørgsmål kun ved at bruge informationen i kilderne.
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

    def generate(self, references: list[(str, str)], messages: list[dict]) -> str:
        # parsed_references = [f"{ref['sentences'][:700]}\nlink: {ref['article_link']}" for ref in references][:3]
        # for ref in references:
        #    print("\nref: ", ref)
        logger.info(f"parsed_references: {references}")

        # remove sources from output if generated
        cleaned_messages = []
        logger.debug("RAW MESSAGES")
        logger.debug(messages)
        logger.debug("END RAW MESSAGES")
        i = 0
        for i, message in enumerate(messages):
            # skip initial welcome message
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

        yield self.llm_generate(
            {
                "messages": cleaned_messages,
                "parameters": {"temperature": 0.1, "max_new_tokens": 600},
            },
            references,
        )

    def llm_format(self, msgs, parsed_references):
        result = "[INST] <<SYS>>\n"
        # if msgs and msgs[0].get("role") == "system":
        #     result += msgs.pop(0).get("content", "")

        result += self.system_message
        result += "\n<</SYS>>[/INST]\n\n"

        if parsed_references:
            for msg in msgs[:-1]:
                result += f"{msg['content']}"
                result += "[INST]" if msg["role"] == "assistant" else "[/INST]"
            result += (
                self.prompt_template
                + "\n\n".join([ref.text for ref in parsed_references])
                + "\n\n"
            )
            result += (
                "\n[INST] Brugerens spørgsmål:"
                + msgs[-1]["content"]
                + "[/INST]Dit svar:"
            )
        else:
            result += self.missing_reference_prompt
        # print(result)
        return result

    def decode(self, input, stream=False):
        try:
            return input.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.debug(f"UnicodeDecodeError: {e}")
            return input.decode("utf-8", errors="ignore")

    def llm_generate(self, input, parsed_references):
        fetch_options = {
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "no-store",
            },
            "method": "POST",
            "redirect": "manual",
        }

        fetch_url = "http://chat-bib-tgi-1-0.mi-prod.svc.cloud.dbc.dk/generate_stream"
        request_body = {
            "inputs": self.llm_format(input["messages"], parsed_references),
            "parameters": input["parameters"],
        }
        request_body_str = json.dumps(request_body)

        res = requests.post(
            fetch_url,
            headers=fetch_options["headers"],
            data=request_body_str,
            stream=True,
        )

        generated_text = ""

        for chunk in res.iter_content(chunk_size=1024):
            if chunk:
                decoded_value = self.decode(chunk, stream=True)
                try:
                    obj = json.loads(decoded_value.replace("data: ", ""))
                    generated_text += obj.get("token", {}).get("text", {})
                    if obj.get("token", {}).get("text", {}) == "</s>":
                        # print("dont print </s>")
                        yield ""
                    else:
                        yield obj.get("token", {}).get("text", {})
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.info(f"Error during streaming: {e}")
