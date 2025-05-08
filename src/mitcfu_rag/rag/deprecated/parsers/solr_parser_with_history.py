#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.solr_parser - solr_parser

============
SolrParser
============

SolrParser preprocesses messages from the chat sent.

example of usage:

    d_parser = SolrParser()
    messages = ["Hej", "  Er der noget om biblioteker?  "]

    prprocessed_messaged = d_parser(messages)
    print(f'preprocessed messaged: {preprocessed_messages}')
"""

import logging
from fakta_chat.rag.rag import Parser

logger = logging.getLogger(__name__)


class SolrParser(Parser):
    def __init__(self):
        pass

    def preprocess(self, messages: list[str]) -> list[str]:
        processed_msg = []
        for message in messages:
            processed_msg.append(f"{message['role']}: {message['content']}\n")

        return processed_msg
