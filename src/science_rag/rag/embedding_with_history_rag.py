#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.solr_rag -- solr_rag model minimum example

============
EmbeddingRAG
============

EmbeddingRAG is a rag model for faktalink.
It takes chat messages as an input and returns a response.

example of usage:
    from fakta_chat.rag.solr_rag import EmbeddingRAG

    d_rag = EmbeddingRAG()
    messages = ["Hej", "Er der noget om miljø?"]

    response = d_rag(messages)
    print(f'response: {response}')
"""

import logging
from typing import Generator, Any
from fakta_chat.rag.rag import RAG
from fakta_chat.rag.parsers.solr_parser import SolrParser
from fakta_chat.rag.retrievers.multilinguale5_large_retriever import EmbeddingRetriever
from fakta_chat.rag.generators.embedding_with_history_generator import (
    EmbeddingGenerator,
)


logger = logging.getLogger(__name__)


class EmbeddingRAG(RAG):
    def __init__(self):
        """
        Components used in the RAG model.
        """
        self.parser = SolrParser()
        self.retriever = EmbeddingRetriever()
        self.reranker = None
        self.generator = EmbeddingGenerator()
        self.validator = None
        self.summarizer = None

    def get_response(self, messages: list[str], *args, **kwargs) -> str:
        """
        Revieves a list of chat messages and returns the next response given by the chatbot.
        """
        # print(messages)
        processed_messages = self.parser(messages)
        similarities, references = self.retriever(messages, n=3)

        if logger.isEnabledFor(logging.DEBUG):
            for i, (similarity, reference) in enumerate(zip(similarities, references)):
                logger.debug(f"{i + 1}. similarity: {similarity:.2f} - {reference}\n")

        generated_answer, generated_sources = self.generator(references, messages)

        if not generated_sources or not generated_answer:
            return "Jeg kan ikke finde svaret på dit spørgsmål. Kan du prøve at stille det på en anden måde?"

        validation = self.validator(
            generated_answer + "\n" + " - ".join(generated_sources),
            references,
            messages,
        )

        # print("VALIDATION: ", validation)

        if validation:
            return generated_answer + "\n" + " - ".join(generated_sources)
        else:
            return "Jeg kan ikke finde svaret på dit spørgsmål. Kan du prøve at stille det på en anden måde?"

    def evaluate(self, messages: list[str]):
        """
        yields response tokens from rag request.
        """

        def gen_wrapper(stream):
            for item in stream:
                for i in item:
                    yield i

        messages = [{"role": "user", "content": messages[0]}]

        similarities, references = self.retriever(messages)

        if logger.isEnabledFor(logging.DEBUG):
            for i, (similarity, reference) in enumerate(zip(similarities, references)):
                logger.debug(f"{i + 1}. similarity: {similarity:.2f} - {reference}\n")

        stream = self.generator(references, messages)
        response = "".join(gen_wrapper(stream))
        return references, response

    def stream_response(
        self, messages: list[dict[str, Any]], *args, **kwargs
    ) -> Generator[str, None, None]:
        """
        yields response tokens from rag request.
        """
        similarities, references = self.retriever(messages)

        # for sim, ref in zip(similarities, references):
        #     print(f"similarity: {sim} - {ref}")

        if logger.isEnabledFor(logging.DEBUG):
            for i, (similarity, reference) in enumerate(zip(similarities, references)):
                logger.debug(f"{i + 1}. similarity: {similarity:.2f} - {reference}\n")

        return references, self.generator(references, messages)
