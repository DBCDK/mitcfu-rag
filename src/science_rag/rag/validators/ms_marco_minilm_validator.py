#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.embedding_retriever - embedding_retriever

============
EmbeddingRetriever
============

EmbeddingRetriever retrieves relevant references based on the messages from the chat sent.
There is no underlying database and EmbeddingRetriever returns an dummy document.

example of usage:
    from fakta_chat.embedding_retriever import EmbeddingRetriever
    d_retriever = EmbeddingRetriever()
    messages = messages = ["Hej", "Er der noget om biblioteker?"]
    refs = d_retriever.retrieve(messages)
    print(f'relevant references: {refs}')
"""

import logging
from mitcfu_rag.rag.rag import Retriever, Reference, Validator
from mitcfu_rag.tools import KNNSearch
from mitcfu_rag.tools.embedder import HuggingfaceEmbedder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
import torch
import torch.nn.functional as F
from os import listdir
import json
from os.path import isfile, join

logger = logging.getLogger(__name__)


class MsValidator(Validator):
    def __init__(self):
        self.device = "cpu"
        self.cross_sentence_model = AutoModelForSequenceClassification.from_pretrained(
            "/data/mitCFU-models/ms-marco-MiniLM-L-6-v2"
        )
        self.cross_sentence_tokenizer = AutoTokenizer.from_pretrained(
            "/data/mitCFU-models/ms-marco-MiniLM-L-6-v2"
        )

    def validate(
        self, generated_response: str, references: list[Reference], query: str
    ):
        pass

    def validate_references(self, query, references, threshold=0.1):
        scores = self.cross_scores([ref.text for ref in references], query)

        for ref in references:
            ref.score = scores.get(ref.text, 0.0)

        filtered_references = [ref for ref in references if ref.score > threshold]
        sorted_filtered_references = sorted(
            filtered_references, key=lambda x: x.score, reverse=True
        )

        return sorted_filtered_references

    # def filter_by_score(self, query, references, threshold=0.1):
    # scores = self.cross_scores([ref.text for ref in references], query)
    # return [ref for ref in references if scores[ref.text] > threshold]

    def cross_scores(self, sentences, query):
        features = self.cross_sentence_tokenizer(
            [query] * len(sentences),
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.cross_sentence_model.eval()
        with torch.no_grad():
            scores = self.cross_sentence_model(**features).logits.flatten()

        return {sentence: float(score) for sentence, score in zip(sentences, scores)}
