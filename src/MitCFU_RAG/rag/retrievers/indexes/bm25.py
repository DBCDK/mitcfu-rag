#!/usr/bin/env python
"""
:mod:`fakta_chat.tools.bm25 -- Embeds texts with bm25 algorithm

========
bm25Embedder
========

BM25Plus algorithm, giving a lower-bound to terms occuring just a single time in long documents: 
http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf

Uses BPE tokenizer.

for usage, see `index_paragraph_fakta_docs` function
"""

__all__ = ['bm25Embedder']

import json
import numpy as np
import pickle
import logging
from tqdm import tqdm

from rank_bm25 import BM25Plus
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from fakta_chat.rag.rag import Reference


logger = logging.getLogger(__name__)

__all__ = ['bm25Embedder', 'BPETokenizer']

PATH_TO_SOLR_INDEX = '/data/faktalink/faktalink-extract-2023-old/index.json'
PATH_TO_BPE = '/data/faktalink/bpe_tokenizer'
PATH_TO_BM25_INDEX = '/data/faktalink/bm25_index'


class bm25Embedder():
    def __init__(self, tokenizer, index, labels):
        self.tokenizer = tokenizer
        self.index = index
        self.labels = labels
        
    def __call__(self, texts: list[str]) -> np.array:
        return self.encode(texts)
    
    @classmethod
    def load(cls, tokenizer, index_path: str, labels_path: str):
        with open(index_path, "rb") as fi:
            bm25_index = pickle.load(fi)
        with open(labels_path, "rb") as fl:
            bm25_labels = pickle.load(fl)
        return cls(tokenizer, bm25_index, bm25_labels)
    
    @classmethod
    def create_index(cls, tokenizer, texts: list[str], 
                     labels: list[Reference]):
        inputs = []
        for doc in texts:
            inputs += [tokenizer.encode(doc)]
        bm25_index = BM25Plus(inputs)
        return cls(tokenizer, bm25_index, labels)
    
    def get_top_n(self, query: str, n: int = 5):
        q_tok = self.tokenizer.encode(query)
        return self.index.get_top_n(query=q_tok, documents=self.labels, n=n)
    
    def get_scores(self, query: str, n: int = 5):
        q_tok = self.tokenizer.encode(query)
        scores = self.index.get_scores(q_tok)
        top_n_scores = np.sort(scores)[::-1][:n]
        return top_n_scores
    
    def save(self, dir_path: str = PATH_TO_BM25_INDEX):
        with open(dir_path + "/index", "wb") as fi:
            pickle.dump(self.index, fi)
        with open(dir_path + "/labels", "wb") as fl:
            pickle.dump(self.labels, fl)


class BPETokenizer():
    def __init__(self, trained_tokenizer) -> None:
        self.tokenizer = trained_tokenizer
    
    @classmethod
    def load(cls, path: str = PATH_TO_BPE):
        return cls(Tokenizer.from_file(path))
    
    @classmethod
    def train(cls, texts: list[str]):
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer()
        tokenizer.train_from_iterator(texts, trainer)
        return cls(tokenizer)
    
    def encode(self, text: str):
        return self.tokenizer.encode(text).tokens
    
    def save(self, dir_path: str):
        self.tokenizer.save(dir_path + '/tokenizer')


def index_paragraph_docs():
    logger.info('Loading documents from Faktalink index file')
    path_to_index_file = PATH_TO_SOLR_INDEX
    
    with open(path_to_index_file, 'r') as file:
        data = json.load(file)
    texts = []
    labels = []
    for doc in tqdm(data):
        for paragraph in doc:
            text = paragraph['subheadline'] + "\n" + " ".join(paragraph['sentences'])
            texts += [text]
            labels += [Reference(id=paragraph['id'],
                                 article_link=paragraph['article_link'],
                                 article_headline=paragraph['article_headline'],
                                 text=text)]
    
    path_to_bpe_tokenizer = PATH_TO_BPE
    logger.info(f'Train BPE tokenizer and save it to disk {path_to_bpe_tokenizer}')
    tokenizer = BPETokenizer.train(texts)
    tokenizer.save(path_to_bpe_tokenizer)

    path_to_bm25_index = PATH_TO_BM25_INDEX
    logger.info(f'Create BM25 index and save it to disk {path_to_bm25_index}')
    bm25_index = bm25Embedder.create_index(tokenizer, texts, labels)
    bm25_index.save()

if __name__ == '__main__':
    index_paragraph_docs()