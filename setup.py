#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- mode: python -*-
from setuptools import setup, find_packages

## See the following pages for keywords possibilities for setup keywords, etc.
# https://packaging.python.org/
# https://docs.python.org/3/distutils/apiref.html
# https://docs.python.org/3/distutils/setupscript.html

setup(
    name="science-rag",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    description="science-rag",
    test_suite="tests",
    # faiss skal tilføjes install_requires, når problemet med faiss er løst.
    install_requires=[
        "langchain",
        "langchain-text-splitters",
        "langchain_community",
        "langchain_unstructured",
        "langgraph",
        "langsmith",
        "streamlit",
        "scikit-learn",
        "numpy",
        "nltk",
        "docling",
        "sentence_transformers",
        "rich",
        "rank_bm25",
        "transformers==4.57.1",
        "torch",
        "dbc_pyutils",
        "tornado",
        "pydantic",
        "aiohttp",
        "asyncio",
        "requests",
        "keybert",
        "tokenizers",
        "accelerate",
    ],
    extras_require={"evaluation": []},
    provides=["science_rag"],
    package_data={"science_rag.evaluation_tool": ["evaluation_tool/data/*"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "create-faiss-index = science_rag.rag.retrievers.indexes.multilinguale5:main",
            "create-faiss-index-generic-parser = science_rag.rag.retrievers.indexes.multilinguale5_generic_parser:main",
            "science-RAG-sh = science_rag.term_ui:cli",
            "evaluate = science_rag.evaluation_tools.evaluation:run",
            "evaluate-retrieval = science_rag.evaluation_tools.evaluate_retrieval:run",
            "compare-retrievers = science_rag.evaluation_tools.compare_retrievers:run",
            "streaming-service-science-rag = science_rag.service:cli",
        ]
    },
    maintainer="ai",
    maintainer_email="ai@dbc.dk",
    zip_safe=False,
)

# Standard setup
# setup(name='science-RAG',
#       version='0.1.0',
#       package_dir={'': 'src'},
#       packages=find_packages(where='src'),
#       description='RAG-løsning til science 2025',
#       test_suite='science_RAG.tests',
#       provides=['science_RAG'],
#       install_requires=[],
#       package_data={'science_RAG': ['data/*', 'data/examples/*', 'data/html/*', 'data/cfg/*']},
#       include_package_data=True,
#       entry_points=
#         {"console_scripts": []},
#       maintainer="ai",
#       maintainer_email="ai@dbc.dk",
#       zip_safe=False)
