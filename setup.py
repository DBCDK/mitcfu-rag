#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- mode: python -*-
from setuptools import setup, find_packages

## See the following pages for keywords possibilities for setup keywords, etc.
# https://packaging.python.org/
# https://docs.python.org/3/distutils/apiref.html
# https://docs.python.org/3/distutils/setupscript.html

setup(
    name="mitcfu-rag",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    description="Chatbot to MitCFU",
    test_suite="tests",
    install_requires=[
        "dbc-data",
        "langchain==0.3.27",
        "langchain-text-splitters==0.3.11",
        "langchain_community>=0.3.31",
        "langchain_unstructured>=0.1.6",
        "langgraph>=0.4.1",
        "streamlit",
        "scikit-learn",
        "numpy",
        "nltk",
        "sentence_transformers",
        "rich",
        "rank_bm25",
        "transformers",
        "torch",
        "dbc_pyutils",
        "tornado",
        "pydantic",
        "aiohttp",
        "requests",
        "tokenizers",
        "accelerate",
    ],
    extras_require={"evaluation": []},
    provides=["mitcfu_rag"],
    package_data={"mitcfu_rag.evaluation_tool": ["evaluation_tool/data/*"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "create-faiss-index = mitcfu_rag.rag.index_vector_db:main",
            "mitcfu-RAG-sh = mitcfu_rag.term_ui:cli",
            "evaluate = mitcfu_rag.evaluation_tools.evaluation:run",
            "evaluate-retrieval = mitcfu_rag.evaluation_tools.evaluate_retrieval:run",
            "compare-retrievers = mitcfu_rag.evaluation_tools.compare_retrievers:run",
            "streaming-service-mitcfu = mitcfu_rag.service:cli",
        ]
    },
    maintainer="ai",
    maintainer_email="ai@dbc.dk",
    zip_safe=False,
)
