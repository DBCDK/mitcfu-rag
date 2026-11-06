#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`mitcfu_rag.retrievers.embedding_retriever - embedding_retriever

============
EmbeddingRetriever
============

EmbeddingRetriever retrieves relevant references based on the messages from the chat sent.
There is no underlying database and EmbeddingRetriever returns an dummy document.

example of usage:
    from mitcfu_rag.embedding_retriever import EmbeddingRetriever
    e_retriever = EmbeddingRetriever()
    messages = messages = ["Hej", "Er der noget om biblioteker?"]
    refs = e_retriever.retrieve(messages)
    print(f'relevant references: {refs}')
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from mitcfu_rag.rag.rag import Reference, Retriever
from mitcfu_rag.tools import KNNSearch
from mitcfu_rag.tools.embedding_factory import create_embedder
from mitcfu_rag.tools.message_history import clean_sources_from_messages

logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = "/data/rani/mitcfu-data/10plus-abstract-77295-jeds-e5-multilingual-instruct-faiss-index"
MODEL_PATH = "/data/mitCFU-models/multilingual-e5-large"
CROSS_MODEL_PATH = "/data/mitCFU-models/ms-marco-MiniLM-L-6-v2"


class EmbeddingRetriever(Retriever):
    def __init__(
        self,
        model_path: str | None = None,
        embeddings_path: str = EMBEDDINGS_PATH,
        cross_model_path: str | None = None,
        jed_document_path: str | None = None,
        *,
        embedding_backend: str | None = None,
        embedder=None,
    ):
        """Construct query embedder and local reranker.

        ``model_path`` is a *local Hugging Face model path*, retained for
        compatibility with AgenticRAG's ``embedding_model`` argument.
        API mode ignores that path and reads EMBEDDING_MODEL from the same
        environment configuration as the indexing script. In local mode,
        model_path overrides LOCAL_EMBEDDING_MODEL_PATH if provided.

        ``embedder`` optionally injects an implementation in tests.
        """
        self.device = "cpu"
        backend = (
            (embedding_backend if embedding_backend is not None else os.getenv("EMBEDDING_BACKEND", "api"))
            .strip()
            .lower()
        )
        self.embedder = embedder or create_embedder(
            backend=backend,
            local_model_path=(model_path or os.getenv("LOCAL_EMBEDDING_MODEL_PATH") or MODEL_PATH)
            if backend == "local"
            else None,
        )
        logger.info("Query embeddings: backend=%s, model=%s", backend, self.embedder.model)

        self.cross_model = AutoModelForSequenceClassification.from_pretrained(cross_model_path, device_map=self.device)
        self.cross_tokenizer = AutoTokenizer.from_pretrained(cross_model_path, device_map=self.device)
        self.cross_model.to(self.device)
        self.cross_model.eval()

        index_dir = Path(embeddings_path)
        self.searcher = KNNSearch.load(str(index_dir / "embeddings"), str(index_dir / "labels.npy"))
        self.jed_document_path = jed_document_path
        if jed_document_path:
            self.all_articles = self.initiate_articles()
            self.all_materialtypes = self.identify_all_materialtypes()
        else:
            self.all_articles = {}
            self.all_materialtypes = set()

        self.validator = None
        self.task = "Given a web search query, retrieve relevant passages that answer the query"

    def initiate_articles(self):
        with open(self.jed_document_path, encoding="utf-8") as file:
            all_documents = json.load(file)
        all_articles = {}
        for document in all_documents:
            for doc_id, doc in document.items():
                if doc.get("abstract"):
                    all_articles[str(doc_id)] = self.format_doc(doc_id, doc)
        return all_articles

    def identify_all_materialtypes(self):
        all_materialtypes = set()
        for article in self.all_articles.values():
            all_materialtypes.update(article.materialtypes)
        return all_materialtypes

    def format_doc(self, doc_id, doc):
        # Work info
        text = " ".join(doc.get("abstract"))
        subjects = [sub.get("display") for sub in doc.get("subjects", {}).get("all", {}).get("subjects", [])]
        material_types_general = [mat.get("general").get("display") for mat in doc.get("materialTypes", [])]
        material_types_specific = [mat.get("general").get("specific") for mat in doc.get("materialTypes", [])]
        languages = [lan.get("display") for lan in doc.get("mainLanguages", [])]
        creators_person = [lan.get("display") for lan in doc.get("creators", {}).get("persons", [])]
        series_titles = [serie.get("title") for serie in doc.get("series", [])]
        manifestation = doc.get("manifestations", {}).get("all")[0]
        creators_publisher = manifestation.get("publisher", [])
        audience_subject = manifestation.get("audience", {}).get("generalAudience", [])
        article_link = "https://mitcfu.dk/MaterialeInfo/?faust=" + str(doc_id)
        if access_urls := manifestation.get("access", {}).get("accessUrls", []):
            for url in access_urls:
                if "https://link.mitcfu.dk/m/" in url.get("url", ""):
                    article_link = url.get("url")
        return Reference(
            id=str(doc_id),
            article_headline=doc.get("titles").get("full")[0],
            article_link=article_link,
            score=0.0,
            text=text,
            chunk="Not chunked",
            keywords=subjects,
            creators=creators_person + creators_publisher,
            audience=audience_subject,
            materialtypes=material_types_general + material_types_specific,
            publicationdate=doc.get("firstPublicationDate", None),
            languages=languages,
            series=series_titles,
        )

    async def async_retrieve(self, messages: dict, n: int = 5, follow_up=True):
        return await self.retrieve(messages, n, follow_up=follow_up)

    async def async_rerank_retrieve(self, messages: dict, n: int = 5, follow_up=True):
        return await self.rerank_retrieve(messages, limit=n, follow_up=follow_up)

    async def retrieve(self, input: dict, n: int = 3, follow_up: bool = False):
        messages = clean_sources_from_messages(input["input"])
        if not messages:
            return [], []
        if follow_up:
            question = ", ".join(message["content"] for message in messages)
        else:
            question = messages[-1]["content"]
        query = f"Instruct: {self.task}\nQuery: {question}"
        logger.debug("Retrieving resources for %s", query)
        return await self.get_docs(query, n)

    async def get_docs(self, query: str, limit: int = 3):
        """Generate one (1, dimensions) vector without blocking the event loop.

        Both API and local backends expose embed_documents() -> float32 NumPy
        rows. This path preserves the existing E5 query instruction prefix.
        """
        if limit <= 0:
            return [], []
        embedded_query = await asyncio.to_thread(self.embedder.embed_documents, [query])
        embedded_query = np.asarray(embedded_query, dtype=np.float32)
        if embedded_query.ndim != 2 or embedded_query.shape[0] != 1:
            raise ValueError(f"Expected one query embedding row, got {embedded_query.shape}")
        if not np.isfinite(embedded_query).all():
            raise ValueError("Query embedding contains NaN or infinity")
        return await self.search(embedded_query, limit)

    async def search(self, embedded_query, limit, filters=None):
        """Map FAISS chunk IDs back to original document references.

        Keep scores and returned references aligned even if filtering removes
        documents; an index/JSON mismatch is logged and skipped.
        """
        if limit <= 0:
            return [], []
        hits = await self.searcher.search(embedded_query, limit * 100)
        if not hits:
            return [], []
        scores, retrieved_articles = [], []
        for chunk_id, score in hits:
            doc_id = str(chunk_id).rsplit("_chunk", maxsplit=1)[0]
            article = self.all_articles.get(doc_id)
            if article is None:
                logger.warning("FAISS returned unknown document id %s", doc_id)
                continue
            if filters and not filtered(article, filters):
                continue
            retrieved_articles.append(article)
            scores.append(score)
            if len(retrieved_articles) == limit:
                break
        return scores, retrieved_articles

    async def rerank_retrieve(self, input: dict, limit: int = 50, follow_up=True):
        queries = input.get("reformulated_queries") or []
        if not queries or limit <= 0:
            return [], []
        all_results = {}
        articleid2article = {}
        logger.debug("Retrieving resources for %s", queries)
        for original_query in queries:
            query = f"Instruct: {self.task}\nQuery: {original_query}"
            _, articles = await self.get_docs(query, limit=limit)
            for article in articles:
                articleid2article[article.id] = article
            scores = await self.cross_select_top_sentences(
                articles, query, limit=max(1, min(limit, 40 // len(queries)))
            )
            all_results[query] = scores

        if len(queries) > 1:
            ranked_ids = list((await reciprocal_rank_fusion(all_results)).keys())
        else:
            # The original implementation incorrectly indexed with the *raw*
            # query and returned ID strings instead of Reference objects.
            ranked_ids = list(next(iter(all_results.values())).keys())
        return [], [articleid2article[doc_id] for doc_id in ranked_ids[:limit]]

    async def cross_select_top_sentences(self, articles, query, limit=50):
        if not articles or limit <= 0:
            return {}

        def score_on_cpu():
            sentences = [art.text for art in articles]
            features = self.cross_tokenizer(
                [query] * len(sentences),
                sentences,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                return self.cross_model(**features).logits.flatten().cpu().numpy()

        # The reranker is synchronous CPU work, so also offload it from the
        # async streaming response path.
        scores = await asyncio.to_thread(score_on_cpu)
        top_indices = np.argsort(-scores)[:limit]
        return {articles[i].id: float(scores[i]) for i in top_indices}


async def reciprocal_rank_fusion(search_results_dict, k=100):
    fused_scores = {}
    for doc_scores in search_results_dict.values():
        for rank, (doc, _score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            fused_scores[doc] = fused_scores.get(doc, 0) + 1 / (rank + k)
    return dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))


def filtered(article, filters):
    """Return True when each field intersects with its allowed values."""
    for field, allowed in filters.items():
        actual = getattr(article, field, None)
        actual_values = actual if isinstance(actual, (list, tuple, set)) else [actual]
        allowed_values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        if not set(actual_values).intersection(allowed_values):
            return False
    return True
