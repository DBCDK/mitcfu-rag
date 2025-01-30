#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.solr_rag -- solr_rag model minimum example

============
SolrRAG
============

SolrRAG is a rag model for faktalink. 
It takes chat messages as an input and returns a response.

example of usage:
    from fakta_chat.rag.solr_rag import SolrRAG

    d_rag = SolrRAG()
    messages = ["Hej", "Er der noget om miljø?"]

    response = d_rag(messages)
    print(f'response: {response}')
"""
import logging
from typing import Generator, Any
from fakta_chat.rag.rag import RAG
#from fakta_chat.rag.parsers.solr_parser import SolrParser
from fakta_chat.rag.parsers.solr_parser_with_history import SolrParser
from fakta_chat.rag.retrievers.solr_retriever import SolrRetriever
#from fakta_chat.rag.generators.solr_generator import SolrGenerator
from fakta_chat.rag.generators.solr_generator_with_history import SolrGenerator
from fakta_chat.rag.validators.solr_validator import SolrValidator
from fakta_chat.rag.summarizers.general_summarizer import GeneralSummarizer


logger = logging.getLogger(__name__)


class SolrRAG(RAG):
    def __init__(self):
        """
        Components used in the RAG model.
        """
        self.parser = SolrParser()
        self.retriever = SolrRetriever()
        self.reranker = None
        self.generator = SolrGenerator()
        self.validator = SolrValidator()
        self.summarizer = GeneralSummarizer()

    def get_response(self, messages: list[str], *args, **kwargs) -> str:
        """
        Revieves a list of chat messages and returns the next response given by the chatbot.
        """
        processed_messages = self.parser(messages)
        similarities, references = self.retriever(processed_messages)
        
        if logger.isEnabledFor(logging.DEBUG):
           for i, (similarity, reference) in enumerate(zip(similarities, references)):
               logger.debug(f'{i+1}. similarity: {similarity:.2f} - {reference}\n')

        
        generated_answer, generated_kilder = self.generator(references, messages)
        
        if not generated_kilder or not generated_answer:
            return "Jeg kan ikke finde svaret på dit spørgsmål. Kan du prøve at stille det på en anden måde?"
        
        validation = self.validator(generated_answer + "\n" + " - ".join(generated_kilder), references, messages)

        print("VALIDATION: ", validation)

        if validation:
            return generated_answer + "\n" + " - ".join(generated_kilder)
        else:
            return "Jeg kan ikke finde svaret på dit spørgsmål. Kan du prøve at stille det på en anden måde?"
        
    def get_summary(self, messages: list[str]) -> str:
        """
        Takes a current summary, query and answer as input and returns a new summary.
        """
        # get latest message from summarizer, starting from end of list
        current_summary = ""
        for msg in reversed(messages):
            if msg["role"] == "summarizer":
                current_summary = msg["content"]
                break
        
        query = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                query = msg["content"]
                break

        answer = ""
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                answer = msg["content"]
                https_index = answer.find("https")
                if https_index != -1:
                    answer = answer[:https_index]
                break
        
        summary = self.summarizer(current_summary, query, answer)
        print("SUMMARY: ", summary)
        return summary
    
    def evaluate(self, messages: list[str]):
        """
        Takes a list of chat messages as input and returns retrieved references given to the generator
        and the generated response for evaluation.
        """
        references = self.retriever.get_references(messages)
        response = self.get_response(messages)
        return references, response
    
    def stream_response(self, messages: list[dict[str, Any]], *args, **kwargs) -> Generator[str, None, None]:
        """
        yields response tokens from rag request.
        """
        pass
