#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.dummy_generator - dummy_generator

============
DummyGenerator
============

DummyGenerator generates an answer based on a list of references and the query.

example of usage:

    d_generator = DummyGenerator()
    query = "Er der noget om biblioteker?"
    references = ['På visse biblioteker kan du låne fiskestænger, så du kan fange din egen middag efter at have læst om det.']
    response = d_generator(references, query)
    print(f'response: {response}')
"""
import random
import logging
from mitcfu_rag.rag.rag import Generator

logger = logging.getLogger(__name__)


class DummyGenerator(Generator):

    def __init__(self):
        pass

    def generate(self, references: list[str], query: str) -> str:
        fillers = ["", "Hmm...", "", "Lad mig se...", "Et øjeblik...", "Vent lige...", "Hmm, lad mig finde noget..."]
        intros = ["Jeg kunne finde de følgende artikler på Faktalink med relevante tekst passager:", 
                "Her er nogle artikler jeg fandt på Faktalink:", 
                "Her er nogle artikler jeg fandt:", 
                "Jeg tænker disse artikler kunne være relevante for dig:"]
        
        if not references:
            response_str = random.choice(fillers) + " " + "Jeg kunne ikke finde nogle relevante artikler på Faktalink. Kan jeg hjælpe dig med noget andre eller kan du finde en andre måde at beskrive hvad du søger?"
        else:
            response_str = f"""
            {random.choice(fillers)} {random.choice(intros)} \n
            """
            for reference in references:
                response_str += f"""
                - {reference}
                """
        
        return response_str
