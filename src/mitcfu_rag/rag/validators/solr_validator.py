#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.solr_validator - solr_validator

=============
SolrValidator
=============

SolrValidator generates an answer based on a list of references and the query.

example of usage:

    d_generator = SolrValidator()
    query = "Er der noget om biblioteker?"
    references = ['På visse biblioteker kan du låne fiskestænger, så du kan fange din egen middag efter at have læst om det.']
    response = d_generator(references, query)
    print(f'response: {response}')
"""

import random
import logging
import json
from huggingface_hub import InferenceClient
from mitcfu_rag.rag.rag import Validator, AnswerWithNumber

logger = logging.getLogger(__name__)


class SolrValidator(Validator):
    def __init__(self):
        self.client = InferenceClient(
            "http://skolegpt-tgi-1-0.mi-prod.svc.cloud.dbc.dk"
        )

    def validate(
        self, generated_answer: str, references: list[(str, str)], messages: str
    ) -> str:
        if not generated_answer:
            return False

        if (
            "http" in generated_answer
            and not "https://faktalink.dk/" in generated_answer
        ):
            return False

        parsed_references = [
            f"Kilde: {ref['subheadline']} tekst: {ref['sentences'][:250]}\n"
            for ref in references
        ][:3]

        http_index = generated_answer.find("https")
        if http_index != -1:
            generated_answer = generated_answer[:http_index]

        # shorten references, but might be bad for the model
        query = messages[-1]["content"]

        # chat_history = ""
        # for msg in reversed(messages):
        #     if msg["role"] == "summarizer":
        #         chat_history = msg["content"]
        #         break

        # if chat_history == "":
        #     chat_history = " ".join([msg["content"][:http_index] for msg in messages[:-1] if (http_index := msg["content"].find("https")) != -1])
        prompt = (
            """
Du er ekspert i at læse FaktaBots svar kritisk igennem, og den har fundet relevante kilder til brugeren.
På en skala fra 1-4, vurder i hvor høj FaktaBots svar er relevant til brugerens forespørgsel.

Du skal vurdere FaktaBots svar ud fra følgende kriterier:
- Svaret får en høj score, hvis det er relevant for brugerens spørgsmål.
- Svar får en høj score, hvis det ikke indeholder stødende, racistisk eller upassende indhold.

Kilder: """
            + "\n".join(parsed_references)
            + """
Brugerens forespørgsel: """
            + query
            + """
FaktaBots svar: """
            + generated_answer
            + """

Giv dit svar som json i denne format:
{
    "score": din score
}
Dit svar:"""
        )

        print("VALIDATION PROMPT")
        print(prompt)
        print("")

        response_str = self.client.text_generation(
            prompt,
            max_new_tokens=33,
            temperature=0.1,
            grammar={"type": "json", "value": AnswerWithNumber.model_json_schema()},
            return_full_text=False,
        )

        try:
            json_response = json.loads(response_str)
        except json.JSONDecodeError:
            return False
        if json_response["score"] >= 3:
            return True
        else:
            return False
