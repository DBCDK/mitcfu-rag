#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`mitcfu_rag.rag.agent_streaming_rag` -- agent_streaming_rag model

============
AgenticRAG
============

AgenticRAG is a rag model for mitcfu.
It takes chat messages as an input and returns a response.

example of usage:
    from mitcfu_rag.rag.agent_streaming_rag import AgenticRAG

    a_rag = AgenticRAG()
    messages = ["Hej", "Er der noget om miljø?"]

    response = a_rag(messages)
    print(f'response: {response}')
"""

import logging
import asyncio
from typing import Generator, Any
from mitcfu_rag.rag.rag import RAG
from mitcfu_rag.tools.llm_formatting import clean_sources_from_messages
# from mitcfu_rag.rag.retrievers.streaming_mistral_retriever import Mistrale5Retriever
from mitcfu_rag.rag.retrievers.streaming_multilingual_retriever import (
    EmbeddingRetriever,
)

from retrieval_utils.embedders import LocalHFEmbedder
from retrieval_utils.retrievers.embedding import EmbeddingRetriever
from retrieval_utils.index.faiss_store import FAISSVectorStore

# from mitcfu_rag.rag.retrievers.multilinguale5_large_retriever import EmbeddingRetriever
from mitcfu_rag.rag.generators.agent_streaming_generator import AgentStreamingGenerator

logger = logging.getLogger(__name__)


class AgenticRAG(RAG):
    def __init__(
        self,
        embedding_model,
        faiss_index,
        jed_document_path,
        validator_model=None,
        use_ceph=False,
    ):
        """
        Components used in the RAG model.
        """
        self.parser = None
        self.embedder = LocalHFEmbedder(
            model_path=embedding_model,
            query_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
            max_length=512,
            device="cpu",
            batch_size=256,
        )
        self.vector_store = FAISSVectorStore.load("test_mitcfu_store")
        self.retriever = EmbeddingRetriever(
            embedder=self.embedder,
            store=self.vector_store,
        )
        self.reranker = None
        self.generator = AgentStreamingGenerator(use_ceph=use_ceph)
        if validator_model:
            self.validator = None
        self.summarizer = None
        self.latest_references = []

    async def stream_response(
        self, input: dict[str, Any], prompt_template, limit=5, *args, **kwargs
    ) -> Generator[str, None, None]:
        """
        yields response tokens from rag request.
        """
        messages = clean_sources_from_messages(input["input"])[-1]["content"]
        if input.get("agent", "") == "RAG":
            if input.get("reformulated_queries"):
                results = await asyncio.gather(
                    self.retriever.retrieve(messages, top_k=limit)
                )
            else:
                results = await asyncio.gather(self.retriever.retrieve(messages))
            references = results[0]
            references = references[:limit]
        elif input.get("agent", "") == "FOLLOW_UP":
            input["FOLLOW_UP"] = True
            if input.get("reformulated_queries"):
                results = await asyncio.gather(
                    self.retriever.retrieve(messages, top_k=limit)
                )
            else:
                results = await asyncio.gather(
                    self.retriever.retrieve(messages)
                )
            references = results[0]
            references = references[:limit]
        else:
            references = None

        print(references)
        async for item in self.generator.generate(references, input, prompt_template):
            yield item

    def get_response(self, messages: list[str], *args, **kwargs) -> str:
        """
        Revieves a list of chat messages and returns the next response given by the chatbot.
        """
        # processed_messages = self.parser(messages)
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
