#!/usr/bin/env python
"""
:mod:`mitcfu_rag.tools.message_history` -- chat-history cleanup utilities

==============
Message History
==============

App-level chat-history massaging shared by the generator and retriever.
"""


def clean_sources_from_messages(messages: list[dict]):
    cleaned_messages = []
    for message in messages:
        if message["role"] == "assistant":
            message["content"] = message["content"].lower().split("**kilder**:")[0]
            cleaned_messages.append(message)
        else:
            cleaned_messages.append(message)
    return cleaned_messages
