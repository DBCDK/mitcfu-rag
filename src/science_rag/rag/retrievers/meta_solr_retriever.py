#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.meta_solr_retriever - solr_retriever soely based on metadata-fields

============
MetaSolrRetriever
============

MetaSolrRetriever retrieves relevant references solely based on matches in metadata-fields
(search does not use "sentences" field / paragraph text content itself).

The underlying database is a solr set up with Dockerfile in https://gitlab.dbc.dk/ai/fakta-chat-solr
and indexed faktalink articles using `index-faktalink-articles` module.

for usage, see `main` function.
"""

import re
import logging
import argparse

from mitcfu_rag.rag.rag import Retriever, Reference
from mitcfu_rag.tools import solr_retrieval_tools
from fakta_chat_solr.solr.search import Searcher
from fakta_chat_solr.keywords_generator import KeywordGenerator
from collections import Counter


logger = logging.getLogger(__name__)

KEYWORD_MODEL_PATH = "/data-nfs/kdd-cup/kddcup2024/models/all-MiniLM-L6-v2"
SOLR_URL = "http://xpdev-p01:8800/solr/fakta-chat-solr"


class MetaSolrRetriever(Retriever):
    def __init__(self, solr_url: str = SOLR_URL, model_path: str = KEYWORD_MODEL_PATH):
        self.searcher = Searcher(solr_url)
        self.keyword_extractor = KeywordGenerator(model_path)

    def retrieve(
        self, messages: list[str], limit: int = 5, cli: bool = False
    ) -> tuple[list[float], list[Reference]]:
        message = messages[-1]["content"]
        all_results = []

        # if the message starts with a question, extract the question and
        # search by matching questions in the db
        questions = re.findall(r"^.*\?", message)
        logger.debug(f"extracted questions: {questions}")
        if len(questions) == 1:
            results = solr_retrieval_tools.search_by_question(
                questions[0], self.searcher, n=5
            )
            if cli:
                print(f'- results by matching question "{questions[0]}": \n')
                for result in results:
                    print(result)
            all_results += results

        # search with full message string in metadata-fields
        all_results += solr_retrieval_tools.search_by_meta(message, self.searcher, n=5)
        if cli:
            print("- results by meta fields: \n")
            for result in results:
                print(result)

        # search with extracted keywords from the message string in metadata-fields
        keywords = self.keyword_extractor.get_1gram_keywords(
            message, n=1
        ) + self.keyword_extractor.get_2gram_keywords(message, n=1)
        logger.debug(f"extracted keywords: {keywords}")
        keyword_search_params = [
            "article_headline^50",
            "subheadline^50",
            "article_topics^100",
            "gen_keywords^50",
        ]
        for keyword in keywords:
            results = solr_retrieval_tools.search(
                keyword, self.searcher, n=5, params=keyword_search_params
            )
            if cli:
                print(f'- results by matching keyword "{keyword}": \n')
                for result in results:
                    print(result)
            all_results += results

        # rerank search results based on the occurences across the different searches
        return self.rerank(all_results, limit=limit)

    def rerank(self, references: list[Reference], limit: int):
        """
        Reranks the references based on the number of occurences in the list.
        """
        count_occurences = Counter(references).most_common(limit)

        for ref, count in count_occurences:
            logger.debug(f"{ref}: {count}")
        return [float(count) for _, count in count_occurences], [
            ref for ref, _ in count_occurences
        ]


def main(args, cli):
    QE = MetaSolrRetriever(args.solr_url)
    message = [{"role": "user", "content": args.message}]
    similarities, references = QE.retrieve(message, cli=cli)
    print(references)
    for i, (sim, ref) in enumerate(zip(similarities, references)):
        print(f"Rank: {i + 1}\n")
        print(f"Similarity: {sim}")
        print(f"{ref} \n")
        print("-----------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--solr_url", type=str)
    parser.add_argument("--message", type=str)
    parser.add_argument(
        "--verbose", help="increase output verbosity", action="store_true"
    )
    args = parser.parse_args()
    cli = False
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        cli = True
    main(args, cli)
