#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.reciprocal_ensembler - ensembles different retrievers with Reciprocal Rank Fusion algorithm

============
ReciprocalEnsembler
============

implements langchain EnsembleRetriever: https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/ensemble/
The EnsembleRetriever takes a list of retrievers as input and ensemble the results of their get_relevant_documents() methods 
and rerank the results based on the simple **Reciprocal Rank Fusion algorithm**.

**Reciprocal Rank Fusion algorithm**:
Reranks a set of documents from different retrievers based on their ranks, 
but does not vanish towards lower-ranked docs due to a set constant c=60 in comparison to e.g. an exponential function.
Is independent of different similarity scores between query and retrieval method.
https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

Implementation follows the langchain example:
https://api.python.langchain.com/en/latest/_modules/langchain/retrievers/ensemble.html#EnsembleRetriever.weighted_reciprocal_rank
"""
import logging
import argparse

from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document
from fakta_chat.rag.rag import Ensembler, Retriever, Reference

from fakta_chat.rag.retrievers.meta_solr_retriever import MetaSolrRetriever
from fakta_chat.rag.retrievers.bm25_retriever import BM25Retriever
from fakta_chat.rag.retrievers.mistrale5_instruct_retriever import Mistrale5Retriever

DEFAULT_RETRIEVERS = [MetaSolrRetriever(), BM25Retriever(), Mistrale5Retriever()]
DEFAULT_WEIGHTS = [1/3, 1/3, 1/3]


class ReciprocalEnsembler(Ensembler):
    def __init__(self, retrievers: list[Retriever] = DEFAULT_RETRIEVERS, 
                 weights: list[float] = DEFAULT_WEIGHTS):
        if sum(weights) != 1:
            raise ValueError("The sum of the weights must be 1")
        self.ensemble_retriever = self.set_up_langchain_ensemble_retriever(weights)
        self.retrievers = retrievers
        self.weights = weights
    
    def set_up_langchain_ensemble_retriever(self, weights, c: int = 60):
        """
        langchain.retrievers.EnsembleRetriever requires a list of retrievers of type
        Runnable[RetrieverInput: str, RetrieverOutput: list[langchain.docstore.document.Document]]
        We just use the EnsembleRetriever.weighted_reciprocal_rank function, 
        which does not use the retrievers attribute.
        """
        return EnsembleRetriever(retrievers=[], weights=weights, c=c)
    
    def ensemble(self, messages: list[str], n: int = 5) -> list[Reference]:
        doc_lists = []
        for retriever in self.retrievers:
            _, relevant_documents = retriever.retrieve(messages)
            doc_lists.append(self.convert_references_to_langchain_docs(relevant_documents))
        ensembled_docs = self.ensemble_retriever.weighted_reciprocal_rank(doc_lists)
        return self.convert_langchain_docs_to_references(ensembled_docs)
    
    def ensemble_by_ranked_docs(self, ref_lists: list[list[Reference]], weights: list[float]) -> list[Reference]:
        doc_lists = [self.convert_references_to_langchain_docs(list_refs) for list_refs in ref_lists]
        ensembled_docs = self.ensemble_retriever.weighted_reciprocal_rank(doc_lists)
        return None, self.convert_langchain_docs_to_references(ensembled_docs)
    
    def convert_references_to_langchain_docs(self, refs: list[Reference]) -> list[Document]:
        langchain_docs = []
        for ref in refs:
            langchain_doc = Document(page_content=ref.text, metadata={"id": ref.id,
                                                                      "article_headline": ref.article_headline,
                                                                      "article_link": ref.article_link})
            langchain_docs.append(langchain_doc)
        return langchain_docs
    
    def convert_langchain_docs_to_references(self, docs: list[Document]) -> list[Reference]:
        refs = []
        for doc in docs:
            ref = Reference(id=doc.metadata["id"],
                            article_headline=doc.metadata["article_headline"],
                            article_link=doc.metadata["article_link"],
                            text=doc.page_content)
            refs.append(ref)
        return refs


def main(args):
    QE = ReciprocalEnsembler()
    _, references = QE.retrieve([args.message])
    for i, ref in enumerate(references):
        print(f'Rank: {i+1}\n')
        print(f'{ref} \n')
        print('-----------------------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--message', type=str)
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main(args)