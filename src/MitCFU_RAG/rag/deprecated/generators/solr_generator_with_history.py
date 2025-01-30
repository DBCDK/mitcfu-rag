#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.solr_generator - solr_generator

============
SolrGenerator
============

SolrGenerator generates an answer based on a list of references and the query.

example of usage:

    d_generator = SolrGenerator()
    query = "Er der noget om biblioteker?"
    references = ['På visse biblioteker kan du låne fiskestænger, så du kan fange din egen middag efter at have læst om det.']
    response = d_generator(references, query)
    print(f'response: {response}')
"""
import random
import logging
import json
from huggingface_hub import InferenceClient
from fakta_chat.rag.rag import Generator
from fakta_chat.rag.utils import SourcesWithScore, AnswerWithSource


logger = logging.getLogger(__name__)


class SolrGenerator(Generator):

    def __init__(self):
        self.client = InferenceClient('http://skolegpt-tgi-1-0.mi-prod.svc.cloud.dbc.dk')

    def generate(self, references: list[(str, str)], messages: list[str]) -> str:
        fillers = ["", "Hmm...", "", "Lad mig se...", "Et øjeblik...", "Vent lige...", "Hmm, lad mig finde noget..."]
        
        if not references:
            return "Kan jeg hjælpe dig med noget andre eller kan du finde en andre måde at beskrive hvad du søger?", []
        
        #parsed_references = [f"# {ref['article_headline']}\n## {ref['subheadline']}\n### {ref['sentences'][:175]}\nlink: {ref['article_link']}" for ref in references][:4]
        parsed_references = [f"{ref['sentences'][:275]}\nLink: {ref['article_link']}" for ref in references][:3]
        query = messages[-1]["content"]

        chat_history = ""
        for msg in reversed(messages):
            if msg["role"] == "summarizer":
                chat_history = msg["content"]
                break
        
        if chat_history == "":
            chat_history = " ".join([msg["content"][:http_index] for msg in messages[:-1] if (http_index := msg["content"].find("https")) != -1])

        prompt = """
Du får en bruger forespørgsel, jeres samtale indtil videre og en liste af kilder. Det er ikke sikkert at nogen af kilderne er relevante for brugerens forespørgsel.

Hvis du kan finde svaret på brugens spørgsmål, så skal du give svaret og linket til den eller de kilder du har brugt.
Hvis du ikke kan finde svaret på brugerens spørgsmål, så skal du skrive "Jeg kan ikke svare på det spørgsmål".

Giv dit svar som json i denne format:
START SVAR
{
    "dit_svar": "DIT SVAR PÅ BRUGERENS SPØRGSMÅL",
    "kilder": ["DEN ELLER DE KILDER DU HAR BRUGT",]
}
SLUT SVAR

samtale indtil videre: """ + chat_history + """
Brugerens forespørgsel: """ + query + """
Kilder: \n""" + "\n".join(parsed_references) + """

Dit svar:
"""

        print("GENERATION PROMPT")
        print(prompt)

        response_str = self.client.text_generation(
            prompt,
            max_new_tokens=400,
            grammar={"type": "json", "value": AnswerWithSource.model_json_schema()},
            temperature=0.1,
            return_full_text=False,
        )

        print("GENERATION RESPONSE")
        print(response_str)
        json_response = ""
        try:
            json_response = json.loads(response_str)
        except:
            print("unable to parse json as is. trying another way.")
        if json_response == "":
            try:
                begin_seq = "START SVAR"
                end_seq = "SLUT SVAR"
                begin_idx = response_str.index(begin_seq)
                end_idx = response_str.index(end_seq)
                response_str = response_str[begin_idx + len(begin_seq):end_idx]
                json_response = json.loads(response_str)
            except:
                return "Jeg kunne ikke forstå dit svar. Kan du prøve at svare igen?", []
        
        generated_answer = json_response.get("dit_svar", "")
        generated_kilder = json_response.get("kilder", [])
        #generated_questions = json_response.get("dit_svar", "")
#         generated_sources = ""
#         for resp in json_response.get("kilder", []):
#             if resp["score"] >= 3:
#                 generated_sources += """\n
# """ + resp["title"] + """
# """ + resp["beskrivelse"] + """
# """ + resp["link"] + """
# """

        return generated_answer, generated_kilder