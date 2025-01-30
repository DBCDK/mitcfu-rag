#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- mode: python -*-
from setuptools import setup, find_packages

## See the following pages for keywords possibilities for setup keywords, etc.
# https://packaging.python.org/
# https://docs.python.org/3/distutils/apiref.html
# https://docs.python.org/3/distutils/setupscript.html

setup(name='mitcfu-rag',
      version='0.1.0',
      package_dir={'': 'src'},
      packages=find_packages(where='src'),
      description='Chatbot to MitCFU',
      test_suite='tests',
      # faiss skal tilføjes install_requires, når problemet med faiss er løst. 
      install_requires=["streamlit", "scikit-learn", "numpy", "nltk", "sentence_transformers", 
                        "rich", "rank_bm25", "transformers", "torch", "dbc_pyutils", "tornado",
                        "pydantic", "aiohttp", "asyncio", "requests", "keybert", "tokenizers"],
      extras_require={'evaluation': []},
      provides=['fakta_chat'],
      package_data={'fakta_chat.evaluation_tool': ['evaluation_tool/data/*']},
      include_package_data=True,
      entry_points={"console_scripts": [
              "faktachat-sh = fakta_chat.term_ui:cli",
              "evaluate = fakta_chat.evaluation_tools.evaluation:run",
              "evaluate-retrieval = fakta_chat.evaluation_tools.evaluate_retrieval:run",
              "compare-retrievers = fakta_chat.evaluation_tools.compare_retrievers:run",
              "streaming-service = fakta_chat.service:cli",
      ]},
      maintainer="ai",
      maintainer_email="ai@dbc.dk",
      zip_safe=False)

#Standard setup
# setup(name='MitCFU-RAG',
#       version='0.1.0',
#       package_dir={'': 'src'},
#       packages=find_packages(where='src'),
#       description='RAG-løsning til MitCFU 2025',
#       test_suite='MitCFU_RAG.tests',
#       provides=['MitCFU_RAG'],
#       install_requires=[],
#       package_data={'MitCFU_RAG': ['data/*', 'data/examples/*', 'data/html/*', 'data/cfg/*']},
#       include_package_data=True,
#       entry_points=
#         {"console_scripts": []},
#       maintainer="ai",
#       maintainer_email="ai@dbc.dk",
#       zip_safe=False)