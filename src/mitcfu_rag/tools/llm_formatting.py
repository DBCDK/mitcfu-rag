#!/usr/bin/env python
"""
:mod:`mitcfu_rag.tools.llm_formatting -- Formatting tools for llm input and output

==============
LLM Formatting
==============

Functions for formatting input for llm's and reading the streamed output from llms.
"""

import json
from transformers import AutoTokenizer

from mitcfu_rag.config import (
    MODEL_MAP,
    GEMMA_3_12B,
    MIXTRAL_8X7B
)


def load_tokenizers(model_names: list[str], use_ceph: bool = False):
    """
    Loads tokenizers for the models used in these endpoints.
    If running this in k8s, load the tokenizers from the ceph mount,
    otherwise load it from huggingface or from cache.
    Make sure you are logged into your huggingface account and have access to the models.
    :param tgi_endpoints:
    :return tokenizers:
    """
    tokenizers = {}
    for model_name in model_names:
        if use_ceph:
            tokenizers[model_name] = AutoTokenizer.from_pretrained(
                f"/data/{model_name}"
            )
        else:
            tokenizers[model_name] = AutoTokenizer.from_pretrained(
                MODEL_MAP[model_name]
            )
    return tokenizers


def gen_wrapper(stream, model_name):
    model_function = select_model_function(model_name)
    for item in stream.iter_content(chunk_size=None, decode_unicode=True):
        decoded_item = __decode(item)
        obj = json.loads(decoded_item.replace("data:", ""))
        yield model_function(obj)


async def async_gen_wrapper(stream, model_name):
    model_function = select_model_function(model_name)
    async for item in stream:
        decoded_item = __decode(item)
        obj = json.loads(decoded_item.replace("data:", ""))
        yield model_function(obj)


def select_model_function(model_name):
    if GEMMA_3_12B in model_name.lower():
        return __gemma_gen_wrapper
    elif MIXTRAL_8X7B in model_name.lower():
        return __mixtral_gen_wrapper
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def tgi_input_format(model_name, request_body):
    if GEMMA_3_12B in model_name.lower():
        return __gemma_tgi_input_format(request_body)
    elif MIXTRAL_8X7B in model_name.lower():
        return __mixtral_tgi_input_format(request_body)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def tgi_output_format(model_name, content):
    if GEMMA_3_12B in model_name.lower():
        return __gemma_tgi_output_format(content)
    elif MIXTRAL_8X7B in model_name.lower():
        return __mixtral_tgi_output_format(content)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def __gemma_tgi_output_format(content):
    return {"choices": [{"delta": {"content": content}}]}

def __mixtral_tgi_output_format(content):
    return {"token": {"text": content}}

def __gemma_tgi_input_format(request_body):
    return {
        "messages": request_body["messages"],
        "model": request_body["model"],
        "stream": request_body["stream"],
        "max_tokens": request_body["max_tokens"]
    }

def __mixtral_tgi_input_format(request_body):
    return {
        "inputs": request_body["messages"][0]["content"],
        "parameters": request_body["parameters"],
    }

def __gemma_gen_wrapper(obj):
    token = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
    if not token == "<eos>":
        return token
    return ""


def __mixtral_gen_wrapper(obj):
    if not obj.get("token", {}).get("text", {}) == "</s>":
        return obj.get("token", {}).get("text", {})
    return ""


def __decode(input):
    try:
        if isinstance(input, str):
            return input
        else:
            return input.decode("utf-8").strip()
    except UnicodeDecodeError:
        return input.decode("utf-8", errors="ignore")
