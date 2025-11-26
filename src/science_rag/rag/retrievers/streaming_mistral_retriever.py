#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.rag.retrievers.mistrale5_instruct_retriever

============
Mistrale5Retriever
============

Mistrale5Retriever retrieves relevant references based on embeddings
embeddings generated with the Mistrale5 model (/indexes/mistrale5.py).

for usage, see `main` function.
"""

import logging
import argparse
import json
import torch
import asyncio
import torch.nn.functional as F
# from langchain_community.vectorstores.faiss import FAISS

from science_rag.rag.rag import Retriever, Reference
from science_rag.tools import KNNSearch
from science_rag.rag.retrievers.indexes.mistrale5_instruct import e5mistralEmbedder  # Old import


logger = logging.getLogger(__name__)

PATH_TO_DOC_EMBEDDINGS = "e5_mistral_instruct_embeddings_faiss_index"
PATH_TO_INDEX = "index.json"


class Mistrale5Retriever(Retriever):
    def __init__(
        self,
        path_to_embedding_model: str,
        path_to_doc_embeddings: str = PATH_TO_DOC_EMBEDDINGS,
        path_to_index_file: str = PATH_TO_INDEX,
    ):
        self.model = e5mistralEmbedder(path_to_embedding_model)
        self.max_length = 4096
        self.knn_searcher = KNNSearch.load(
            path_to_doc_embeddings + "/embeddings",
            path_to_doc_embeddings + "/labels.npy",
        )
        self.index2references = self.load_index2references(path_to_index_file)

    def load_index2references(self, path_to_index_file):
        index2references = {}
        with open(path_to_index_file, "r") as file:
            data = json.load(file)
        for doc in data:
            for paragraph in doc:
                text = text = paragraph["subheadline"] + "\n" + " ".join(paragraph["sentences"])
                index2references[paragraph["id"]] = Reference(
                    id=paragraph["id"],
                    article_link=paragraph["article_link"],
                    article_headline=paragraph["article_headline"],
                    text=text,
                )
        return index2references

    def retrieve(self, messages: list[str], n: int = 5) -> tuple[list[float], list[Reference]]:
        # message = messages[-1]['content']
        message = ""
        for msg in messages:
            if msg["role"] == "user":
                message += msg["content"]
        qv = self.model.encode_query(message).numpy()
        top_n = self.knn_searcher.search(qv, n)
        indexes, scores = zip(*top_n)
        return scores, [self.index2references[i] for i in indexes]

    async def async_retrieve(self, messages: list[str], n: int = 5):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.retrieve(messages, n))

    def cosine_similarity(self, queries_emb: list, docs_emb):
        i = len(queries_emb)
        embeddings = F.normalize(torch.cat([queries_emb, docs_emb]), p=2, dim=1)
        scores = (embeddings[:i] @ embeddings[i:].T) * 100
        return scores.tolist()


def main(args):
    QE = Mistrale5Retriever()
    message = [{"role": "user", "content": args.message}]
    similarities, references = QE.retrieve(message)
    for i, (sim, ref) in enumerate(zip(similarities, references)):
        print(f"Rank: {i + 1}\n")
        print(f"Similarity: {sim}")
        print(f"{ref} \n")
        print("-----------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--message", type=str)
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main(args)
