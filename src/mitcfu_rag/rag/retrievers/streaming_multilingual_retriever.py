#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`mitcfu_rag.embedding_retriever - embedding_retriever

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

from aiohttp import request
from dbc_pyutils import setup_logging
import logging
import asyncio
import os
from collections import defaultdict
from mitcfu_rag.rag.rag import Retriever, Reference
from mitcfu_rag.rag.validators.ms_marco_minilm_validator import MsValidator
from mitcfu_rag.tools import KNNSearch
import requests
from mitcfu_rag.tools.embedder import HuggingfaceEmbedder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
import torch
import torch.nn.functional as F
import aiohttp
from os import listdir
import json
from os.path import isfile, join

logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = "/data/rani/mitcfu-data/10plus-abstract-77295-jeds-e5-multilingual-instruct-faiss-index"
MODEL_PATH = "/data/mitCFU-models/multilingual-e5-large"


class EmbeddingRetriever(Retriever):
    def __init__(
        self,
        model_path=MODEL_PATH,
        embeddings_path=EMBEDDINGS_PATH,
        jed_document_path=None,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_path, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, device_map=self.device
        )
        self.model.to(self.device)
        self.searcher = KNNSearch.load(
            embeddings_path + "/embeddings", embeddings_path + "/labels.npy"
        )
        if jed_document_path:
            self.jed_document_path = jed_document_path
            self.all_articles = self.initiate_articles()
        else:
            self.all_articles = {}
        self.validator = None
        self.session = aiohttp.ClientSession()

    def initiate_articles(self):
        with open(self.jed_document_path) as f:
            all_documents = json.load(f)
        all_articles = {}

        for doc in all_documents:
            for id, content in doc.items():
                text = content.get("abstract")
                if text:
                    all_articles[str(id)] = Reference(
                        id=str(id),
                        article_headline=content.get("titles").get("full"),
                        article_link="https://mitcfu.dk/MaterialeInfo/?faust=" + str(id),
                        score=0.0,
                        text=text[0],
                        chunk="Not chunked",
                    )

        return all_articles

    async def async_retrieve(self, messages: list[str], n: int = 5):
        # loop = asyncio.get_running_loop()
        return await self.retrieve(messages, n)

    async def retrieve(self, input: list[str], n: int = 3):
        # query = f"query: {' '.join([message['content'] for message in messages if message['role'] == 'user'])}"
        messages = input["input"]
        query = f"query: {messages[-1]['content']}"
        return await self.get_docs(query, n)

    # https://huggingface.co/intfloat/multilingual-e5-large
    async def get_docs(self, query: str, limit: int = 3):
        model_device = next(self.model.parameters()).device
        batch_dict = self.tokenizer(
            query, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        batch_dict = {k: v.to(model_device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        embedded_query = (
            F.normalize(embeddings, p=2, dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        hits = await self.searcher.search(embedded_query, limit)
        ids, scores = zip(*hits)
        logger.info(ids)
        #retrieved_articles = [
        #    Reference(id=_id, article_headline="", article_link="", score=0.0, text="")
        #    for _id in ids
        #]

        retrieved_articles = []
        for id, score in zip(ids, scores):
            if "_chunk" in id:
                mitcfu_id, chunk_number = id.rsplit("_chunk", maxsplit=1)
            else:
                mitcfu_id = id
                chunk_number = None

            new_ref = Reference(
                id=mitcfu_id, chunk=chunk_number, article_headline="", article_link=id, score=0.0, text=""
            )
            retrieved_articles.append(new_ref)
            # TODO use the below code instead to actually get the article data
            #if mitcfu_id in self.all_articles:
                #print("processing chunk")
                #mitcfu_ref = self.all_articles[mitcfu_id]
                #print("ref", mitcfu_ref)
                # new_ref = Reference(
                #     id=id,
                #     article_headline=mitcfu_ref.article_headline,
                #     article_link=mitcfu_ref.article_link,
                #     score=score,
                #     text=mitcfu_ref.text,
                #     chunk=f"chunk{chunk_number}" if chunk_number else "Not chunked",
                # )
                #retrieved_articles.append(new_ref)
        # retrieved_articles = [self.all_articles[id] for id in ids if id in self.all_articles]
        return list(scores), retrieved_articles


# [self.all_articles[int(i)] for i in indexes]


def average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
