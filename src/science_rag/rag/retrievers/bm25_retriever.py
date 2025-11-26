#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.rag.retrievers.bm25_retriever - bm25plus on full text

============
BM25Retriever
============

BM25Retriever retrieves relevant references solely based on matches between query string
and a reference's content (Reference.text field).

for usage, see `main` function.
"""

import logging
import argparse

from mitcfu_rag.rag.retrievers.indexes.bm25 import bm25Embedder, BPETokenizer
from mitcfu_rag.rag.rag import Retriever, Reference


logger = logging.getLogger(__name__)

TOKENIZER_PATH = "/data/faktalink/bpe_tokenizer/tokenizer"
# text field contains here "subheadline + sentences of a paragraph, based on data/faktalink/solr_index"
INDEX_PATH = "/data/faktalink/bm25_index/index"
# labels contain Reference objects, based on data/faktalink/solr_index
LABELS_PATH = "/data/faktalink/bm25_index/labels"


class BM25Retriever(Retriever):
    def __init__(
        self,
        tokenizer_path: str = TOKENIZER_PATH,
        index_path: str = INDEX_PATH,
        labels_path: str = LABELS_PATH,
    ):
        tokenizer = BPETokenizer.load(tokenizer_path)
        self.bm25_index = bm25Embedder.load(tokenizer, index_path, labels_path)

    def retrieve(
        self, messages: list[str], n: int = 5
    ) -> tuple[list[float], list[Reference]]:
        message = messages[-1]["content"]
        return self.bm25_index.get_scores(message, n), self.bm25_index.get_top_n(
            message, n
        )


def main(args):
    QE = BM25Retriever()
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
    parser.add_argument(
        "--verbose", help="increase output verbosity", action="store_true"
    )
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main(args)
