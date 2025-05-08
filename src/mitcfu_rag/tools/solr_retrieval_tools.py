#!/usr/bin/env python
"""
:mod:`fakta_chat.tools.solr_retrieval_tools - solr_retrieval_tools

============
Solr Retrieval Tools
============

Tools to transform user requests into solr querries
and return relevant solr documents.
The underlying solr is set up with the Dockerfile in https://gitlab.dbc.dk/ai/fakta-chat-solr
and indexed faktalink articles using `index-faktalink-articles` module.

example of usage:
    import re
    from fakta_chat.tools import solr_retrieval_tools
    from fakta_chat_solr.solr.search import Searcher
    from fakta_chat_solr.keywords_generator import KeywordGenerator

    message = "Hvad er spiseforstyrrelser?"

    keyword_model = KeywordGenerator("/data-nfs/kdd-cup/kddcup2024/models/all-MiniLM-L6-v2")
    keyword = keyword_model.get_1gram_keywords(message, n = 1)[0]
    print(keyword)

    question = re.findall(r'^.*\?', str(message))[0]
    print(question)

    searcher = Searcher("http://localhost:8800/solr/fakta-chat-solr")

    all_refs = solr_retrieval_tools.search(message, searcher, 3)
    question_refs = solr_retrieval_tools.search_by_question(question, searcher, 3)
    keyword_refs = solr_retrieval_tools.search_by_keyword(keyword, searcher, 3)
"""

from fakta_chat_solr.solr.search import Searcher
from mitcfu_rag.rag.rag import Reference

default_search_params = [
    "article_headline^50",
    "article_topics^50",
    "subheadline^100",
    "text",
]
keyword_search_params = ["article_topics^100", "gen_keywords^50"]
question_search_params = ["subheadline^100", "gen_questions^100"]
meta_search_params = ["article_headline", "article_topics", "subheadline"]


def search(
    message: str, searcher: Searcher, n: int, params: list = default_search_params
) -> list[Reference]:
    """
    Takes the whole message or another input and execute a search in solr with the given parameters.
    This method finds matches with all fields given in the parameters.
    """
    results = searcher.solr_search(message, params, limit=n)

    references = [
        Reference(
            ref["id"],
            ref["article_headline"],
            ref["article_link"],
            " ".join(ref["sentences"]),
        )
        for ref in results
    ]
    references = list(set(references))
    return references


def search_by_keyword(
    keyword: str, searcher: Searcher, n: int, params: list = keyword_search_params
) -> list[Reference]:
    """
    Takes the given keywords and execute a search in solr with keyword_search_parameters.
    This method finds matches with article topics and automatic extracted keywords from paragraphs.
    """
    references = []

    results = searcher.solr_search(keyword, params, limit=n)
    references += [
        Reference(
            ref["id"],
            ref["article_headline"],
            ref["article_link"],
            " ".join(ref["sentences"]),
        )
        for ref in results
    ]

    references = list(set(references))
    return references


def search_by_question(
    question: str, searcher: Searcher, n: int, params: list = question_search_params
) -> list[Reference]:
    """
    Takes a questions and execute a search in solr with question_search_parameters.
    This method finds matches with subheadlines of articles formulated in question format
    and automatic generated questions to paragraphs.
    """
    references = []

    results = searcher.solr_search(question, params, limit=n)
    references = [
        Reference(
            ref["id"],
            ref["article_headline"],
            ref["article_link"],
            " ".join(ref["sentences"]),
        )
        for ref in results
    ]

    references = list(set(references))
    return references


def search_by_meta(
    message: str, searcher: Searcher, n: int, params: list = meta_search_params
) -> list[Reference]:
    """
    Takes a message and execute a search in solr with meta_search_parameters.
    This method finds matches with headline, subheadlines and topics of articles.
    """
    references = []

    results = searcher.solr_search(message, params, limit=n)
    references = [
        Reference(
            ref["id"],
            ref["article_headline"],
            ref["article_link"],
            " ".join(ref["sentences"]),
        )
        for ref in results
    ]

    references = list(set(references))
    return references
