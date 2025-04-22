#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.dummy_parser - dummy_parser

============
DummyParser
============

DummyParser preprocesses messages from the chat sent.

example of usage:

    d_parser = DummyParser()
    messages = ["Hej", "  Er der noget om biblioteker?  "]

    prprocessed_messaged = d_parser(messages)
    print(f'preprocessed messaged: {preprocessed_messages}')
"""
import logging
from mitcfu_rag.rag.rag import Parser

logger = logging.getLogger(__name__)


class DummyParser(Parser):
    def __init__(self):
        pass

    def preprocess(self, messages: list[str]) -> list[str]:
        cleaned_messages = [message.strip() for message in messages]
        return cleaned_messages
