#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.dummy_rag -- dummy_rag model minimum example

============
DummyRAG
============

DummyRAG is a rag model for faktalink.
It takes chat messages as an input and returns a response.

example of usage:
    from fakta_chat.rag.dummy_rag import DummyRAG

    d_rag = DummyRAG()
    messages = ["Hej", "Er der noget om miljø?"]

    response = d_rag(messages)
    print(f'response: {response}')
"""

from typing import Generator, Any
import requests
import json
import logging
from fakta_chat.rag.rag import RAG
from fakta_chat.rag.parsers.dummyparser import DummyParser
from fakta_chat.rag.retrievers.dummy_retriever import DummyRetriever
from fakta_chat.rag.generators.dummy_generator import DummyGenerator

logger = logging.getLogger(__name__)


class DummyRAG(RAG):
    def __init__(self):
        """
        Components used in the RAG model.
        """
        self.parser = DummyParser()
        self.retriever = DummyRetriever()
        self.reranker = None
        self.generator = DummyGenerator()
        self.validator = None

    def get_response(self, messages: dict[str, Any]) -> str:
        """
        Revieves a list of chat messages and returns the next response given by the chatbot.
        """
        processed_messages = self.parser(messages[-1]["content"])
        similarities, references = self.retriever(processed_messages)

        if logger.isEnabledFor(logging.DEBUG):
            for i, (similarity, reference) in enumerate(zip(similarities, references)):
                logger.debug(f"{i + 1}. similarity: {similarity:.2f} - {reference}\n")

        response = self.generator(references, processed_messages)
        return response

    def stream_response(
        self, messages: list[dict[str, Any]]
    ) -> Generator[str, None, None]:
        """
        yields response tokens from rag request.
        """
        url = "http://chat-bib-tgi-1-0.mi-prod.svc.cloud.dbc.dk"
        data = {"model": "tgi", "messages": messages, "stream": True, "max_tokens": 200}

        response = requests.post(
            f"{url.rstrip()}/v1/chat/completions",
            data=json.dumps(data),
            stream=True,
            headers={"Content-Type": "application/json"},
        )

        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                data = str(chunk, encoding="utf-8").strip("data: ")
                token = json.loads(data)["choices"][0]["delta"]["content"]
                yield token

    def evaluate(self, messages: list[str]):
        """
        Takes a list of chat messages as input and returns retrieved references given to the generator
        and the generated response for evaluation.
        """
        processed_messages = self.parser(messages[-1])
        similarities, references = self.retriever(processed_messages)

        if logger.isEnabledFor(logging.DEBUG):
            for i, (similarity, reference) in enumerate(zip(similarities, references)):
                logger.debug(f"{i + 1}. similarity: {similarity:.2f} - {reference}\n")

        response = self.generator(references, processed_messages)

        return references, response
