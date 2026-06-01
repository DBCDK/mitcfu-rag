#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.solr_generator - solr_generator

============
SolrGenerator
============

SolrGenerator generates an answer based on a list of references and the query.

This generator is outdated, and should be updated with some of the techniques used in the embedding_with_history_generator

example of usage:

    d_generator = SolrGenerator()
    query = "Er der noget om biblioteker?"
    references = ['På visse biblioteker kan du låne fiskestænger, så du kan fange din egen middag efter at have læst om det.']
    response = d_generator(references, query)
    print(f'response: {response}')
"""

import logging
import json
import requests
from mitcfu_rag.rag.rag import Generator


logger = logging.getLogger(__name__)


class SolrGenerator(Generator):
    def __init__(self):
        self.system_message = """
Du er FaktaChat, en meget klog og kritisk chatbot der KUN bruger FaktaLink kilder til at besvare brugerens spørgsmål.
Du får en bruger forespørgsel og en liste af kilder. Det er ikke sikkert at nogen af kilderne er relevante for brugerens forespørgsel.
Du svarer aldrig på spørgsmål, hvor du ikke kan finde svaret i kilderne.
Det er bedre at sige "Det kan jeg ikke finde svaret på. Prøv at spørge på en anden måde, eller spørg om noget andet.", end det er at opfinde svar eller kilder.

Du opfinder aldrig kilder
Du henviser aldrig til andre links på faktachat end dem du får i forespørgslen.

Dit svar skal bestå af to dele:
Dit svar.
Kilder: En liste af kilder, som du har brugt til at finde svaret.

Kilder: \nKILDE: """

    def generate(self, references: list[(str, str)], messages: list[dict]) -> str:
        parsed_references = [
            f"{ref.text[:700]}\nlink: {ref.article_link}" for ref in references
        ][:3]
        # parsed_references = [f"{ref['sentences'][:700]}\nlink: {ref['article_link']}" for ref in references][:3]
        logger.info(f"parsed_references: {parsed_references}")

        yield self.llm_generate(
            {
                "messages": messages,
                "parameters": {"temperature": 0.1, "max_new_tokens": 600},
            },
            parsed_references,
        )

    def llm_format(self, msgs, parsed_references):
        result = "<s>[INST] <<SYS>>\n"
        # if msgs and msgs[0].get("role") == "system":
        #     result += msgs.pop(0).get("content", "")

        result += self.system_message + "\nKILDE: ".join(parsed_references)
        result += "\n<</SYS>>\n\n"

        for msg in msgs:
            result += f"\n{msg['role']}: {msg['content']}"
            result += "</s><s>[INST]" if msg["role"] == "assistant" else "[/INST]"

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
                    yield obj.get("token", {}).get("text", {})
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.info(f"Error during streaming: {e}")
