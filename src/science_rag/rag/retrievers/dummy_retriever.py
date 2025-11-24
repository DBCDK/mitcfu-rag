#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.dummy_retriever - dummy_retriever

============
DummyRetriever
============

DummyRetriever retrieves relevant references based on the messages from the chat sent.
There is no underlying database and DummyRetriever returns an dummy document.

example of usage:

    d_retriever = DummyRetriever()
    messages = ["Hej", "Er der noget om biblioteker?"]
    sim, refs = d_retriever.retrieve(messages)
    print(f'relevant references: {refs}')
"""

DUMMY_DOCS = [
    "Dummy article: På visse biblioteker kan du låne fiskestænger, så du kan fange din egen middag efter at have læst om det.",
    "$Tøii",
]

import logging
from mitcfu_rag.rag.rag import Retriever

logger = logging.getLogger(__name__)


class DummyRetriever(Retriever):
    def __init__(self):
        pass

    def retrieve(self, messages: list[str]):
        query = self.messages_to_query(messages)
        similarities = [1.0, 0.0]

        relevant_references = []
        relevant_similarities = []
        for i, (similarity, reference) in enumerate(zip(similarities, DUMMY_DOCS)):
            if similarity > 0.0:
                relevant_references.append(reference)
                relevant_similarities.append(similarity)

        return relevant_similarities, relevant_references

    def messages_to_query(self, messages: list[str]) -> str:
        return " ".join(messages)
