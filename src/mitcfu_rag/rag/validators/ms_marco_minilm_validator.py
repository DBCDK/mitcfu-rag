#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`mitcfu_rag.rag.validators.ms_marco_minilm_validator` -- ms-marco cross-encoder validator

=============
MsValidator
=============

MsValidator scores and filters retrieved references against the user's query
using the `ms-marco-MiniLM-L-6-v2` cross-encoder model, so only references
relevant enough to pass a similarity threshold are kept.

example of usage:
    validator = MsValidator()
    filtered_references = validator.validate_references(query, references)
"""

import logging
from mitcfu_rag.rag.rag import Reference, Validator

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

logger = logging.getLogger(__name__)


class MsValidator(Validator):
    def __init__(self):
        self.device = "cpu"
        self.cross_sentence_model = AutoModelForSequenceClassification.from_pretrained(
            "/data/mitCFU-models/ms-marco-MiniLM-L-6-v2"
        )
        self.cross_sentence_tokenizer = AutoTokenizer.from_pretrained("/data/mitCFU-models/ms-marco-MiniLM-L-6-v2")

    def validate(self, generated_response: str, references: list[Reference], query: str):
        pass

    def validate_references(self, query, references, threshold=0.1):
        scores = self.cross_scores([ref.text for ref in references], query)

        for ref in references:
            ref.score = scores.get(ref.text, 0.0)

        filtered_references = [ref for ref in references if ref.score > threshold]
        sorted_filtered_references = sorted(filtered_references, key=lambda x: x.score, reverse=True)

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
