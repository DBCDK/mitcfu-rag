#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`mitcfu_rag.solr_rag -- solr_rag model minimum example

============
EmbeddingRAG
============

EmbeddingRAG is a rag model for mitcfu_rag.
It takes chat messages as an input and returns a response.

example of usage:
    from fakta_chat.rag.solr_rag import EmbeddingRAG

    d_rag = EmbeddingRAG()
    messages = ["Hej", "Er der noget om miljø?"]

    response = d_rag(messages)
    print(f'response: {response}')
"""

import logging
import datetime
import asyncio
import json
from typing import Generator, Any
from mitcfu_rag.rag.rag import RAG, Reference

# from fakta_chat.rag.parsers.solr_parser import SolrParser
# from mitcfu_rag.rag.retrievers.solr_retriever import SolrRetriever
# from mitcfu_rag.rag.retrievers.bm25_retriever import BM25Retriever
# from mitcfu_rag.rag.retrievers.streaming_mistral_retriever import Mistrale5Retriever
from mitcfu_rag.rag.retrievers.streaming_multilingual_retriever import EmbeddingRetriever
from mitcfu_rag.rag.generators.streaming_with_history_generator import EmbeddingGenerator
from mitcfu_rag.rag.validators.ms_marco_minilm_validator import MsValidator


logger = logging.getLogger(__name__)


class StreamingRAG(RAG):
    def __init__(
        self, embedding_model, embeddings_path, jed_document_path, validator_model=None
    ):  # for now, since we are using the smaller model, these arguments are not used: embedding_model, faiss_index, article_index.
        """
        Components used in the RAG model.
        """
        self.parser = None
        self.retriever = EmbeddingRetriever(embedding_model, embeddings_path, jed_document_path)
        # Mistrale5Retriever(embedding_model, faiss_index, article_index)                                 
        self.reranker = None
        self.generator = EmbeddingGenerator()
        self.validator = MsValidator() if validator_model else None
        self.summarizer = None

    def get_response(self, messages: list[str], *args, **kwargs) -> str:
        """
        Receives a list of chat messages and returns the next response given by the chatbot.
        """
        # processed_messages = self.parser(messages)
        similarities, references = self.retriever(messages, n=3)

        if logger.isEnabledFor(logging.DEBUG):
            for i, (similarity, reference) in enumerate(zip(similarities, references)):
                logger.debug(f"{i + 1}. similarity: {similarity:.2f} - {reference}\n")

        generated_answer, generated_sources = self.generator(references, messages)

        if not generated_sources or not generated_answer:
            return "Jeg kan ikke finde svaret på dit spørgsmål. Kan du prøve at stille det på en anden måde?"

        validation = self.validator(generated_answer + "\n" + " - ".join(generated_sources), references, messages)

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

    async def stream_response(
        self, messages: list[dict[str, Any]], version=None, *args, **kwargs
    ) -> Generator[str, None, None]:
        """
        yields response tokens from rag request.
        """
        results = await asyncio.gather(self.retriever.async_retrieve(messages, n=3))
        similarities, references = results[0]
        similarities = similarities[:3]
        references = references[:3]

        async for item in self.generator.generate(references, messages, version):
            yield item
    
    async def stream_response_with_validator(
            self, messages: list[dict[str, Any]], version=None, *args, **kwargs
    ) -> Generator[str, None, None]:
        '''
        Yields response tokens from rag request with validation from MsValidator.
        '''
        results = await asyncio.gather(self.retriever.async_retrieve(messages, n=3))
        similarities, references = results[0]
        similarities = similarities[:3]
        references = references[:3]

        # Validation step. First, we check if the class has the validator attribute.
        if self.validator:
            user_query = messages[-1]["content"] #using the latest user query under the content key
            valid_references = self.validator.validate_references(user_query, references)
            if not valid_references: #If no valid references are found, we tell the user. 
                fallback = {
                    "token": {"text": "Jeg kan ikke finde svaret på dit spørgsmål. Kan du prøve at stille det på en anden måde?"}
                }
                yield f"data: {json.dumps(fallback)}\n\n"
                return
            else:
                references = valid_references
                logger.debug(f"References after validation: {references}")
        
        async for item in self.generator.generate(references, messages, version):
            yield item
