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
import asyncio
from collections import defaultdict
from mitcfu_rag.rag.rag import Retriever, Reference
from mitcfu_rag.rag.validators.ms_marco_minilm_validator import MsValidator
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

path_to_embeddings = "/data/rani/mitcfu-data/10plus-abstract-77295-jeds-e5-multilingual-instruct-faiss-index/embeddings"
path_to_labels = "/data/rani/mitcfu-data/10plus-abstract-77295-jeds-e5-multilingual-instruct-faiss-index/labels.npy"
path_to_JEDs = "/data/rani/mitcfu-data/10plus-abstract-77295-jeds"


class EmbeddingRetriever(Retriever):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            "/data/mitCFU-models/multilingual-e5-large", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/data/mitCFU-models/multilingual-e5-large", device_map="auto"
        )
        #self.model.to(self.device)
        self.searcher = KNNSearch.load(
            path_to_embeddings,
            path_to_labels,
        )
        self.all_articles = self.initiate_articles()
        self.validator = MsValidator()

    def initiate_articles(self):
        article_folder = path_to_JEDs
        onlyfiles = [f for f in listdir(article_folder) if isfile(join(article_folder, f))]
        all_articles = {}

        for file in onlyfiles:
            if ".json" in file and file != "index.json":
                with open(article_folder + "/" + file, "r") as f:
                    article = json.load(f)
                    for id, content in article.items():
                        text = content.get("abstract")
                        if text:
                            all_articles[str(id)] = Reference(
                                id=str(id),
                                article_headline=content.get("titles").get("full"),
                                article_link="MitCFU-ID:" + str(id),
                                score = 0.0,
                                text=text[0],
                            )

        return all_articles

    async def async_retrieve(self, messages: list[str], n: int = 5):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.retrieve(messages, n))

    def retrieve(self, messages: list[str], n: int = 3):
        # query = f"query: {' '.join([message['content'] for message in messages if message['role'] == 'user'])}"
        query = f"query: {messages[-1]['content']}"
        return self.get_docs(query, n)

    # https://huggingface.co/intfloat/multilingual-e5-large
    def get_docs(self, query: str, limit: int = 3):
        model_device = next(self.model.parameters()).device
        batch_dict = self.tokenizer(query, max_length=512, padding=True, truncation=True, return_tensors="pt")
        batch_dict = {k: v.to(model_device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embbeded_query = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy().astype(np.float32)
        hits = self.searcher.search(embbeded_query, limit)
        ids, scores = zip(*hits)
        print(ids)
        retrieved_articles = [self.all_articles[id] for id in ids if id in self.all_articles]
        return list(scores), retrieved_articles


# [self.all_articles[int(i)] for i in indexes]


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
