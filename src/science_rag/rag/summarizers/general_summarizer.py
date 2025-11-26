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
from fakta_chat.rag.rag import Summarizer, SourcesWithScore, AnswerWithSource


logger = logging.getLogger(__name__)


class GeneralSummarizer(Summarizer):
    def __init__(self):
        self.client = InferenceClient("http://skolegpt-tgi-1-0.mi-prod.svc.cloud.dbc.dk")

    def summarize(self, current_summary: str, query: str, answer: str) -> str:
        prompt = (
            """
Lav et løbende resume af samtalen, og tilføj det nyeste spørgsmål og svar til resumeet.
Forsøg at forkorte det nuværende resume hver gang.

START EKSEMPEL

Nuværende resume:
q: hvornår grundloven blev underskrevet. a: grundloven blev underskrevet den 5. juni 1849.

Nye spørgsmål og svar:
Bruger: Hvad skete der ellers i den periode?
AI: I perioden omkring 1849 skete flere betydningsfulde begivenheder:
Treårskrigen (1848-1850) mod Slesvig-Holsten og Preussen.
Overgangen fra enevælde til konstitutionelt monarki i Danmark.

Nyt resume:
q: hvornår grundloven blev underskrevet. a: grundloven blev underskrevet den 5. juni 1849.
q: hvad der ellers skete i den periode. a: i perioden omkring 1849 skete der flere betydningsfulde begivenheder,
herunder Treårskrigen og overgangen fra enevælde til konstitutionelt monarki.

SLUT EKSEMPEL

nuværende resume: """
            + current_summary
            + """
Nye spørgsmål og svar: """
            + query
            + """\n"""
            + answer
            + """

Nyt resume:
"""
        )
        new_prompt = (
            """
Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.
The new summary should contain the essence of the current summary plus a summary of the new lines of conversation.
Write your summary in Danish.

Current summary: + """
            + current_summary
            + """

New lines of conversation: """
            + query
            + """\n"""
            + answer
            + """

New summary:
"""
        )

        response_str = self.client.text_generation(
            prompt,
            max_new_tokens=300,
            temperature=0.1,
            return_full_text=False,
        )

        return response_str.strip()
