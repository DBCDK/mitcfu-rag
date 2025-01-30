#!/usr/bin/env python
"""
:mod:`fakta_chat.tools.mixtrale5_instruct -- Embeds texts with e5-mistral-7b-instruct

========
e5mistralEmbedder
========

e5mistralEmbedder embeds faktalink articles with e5-mistral-7b-instruct model from Huggingface.
e5-mistral-7b-instruct ranks high on ScanEval for Danish (total: 61.7) and can encode 4096 tokens.
https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/
Embeddings can be stored as FAISS db with for fast retrieval.
Implementation from:
https://huggingface.co/intfloat/e5-mistral-7b-instruct

for usage, see `index_paragraph_fakta_docs` function
"""

__all__ = ['e5mistralEmbedder']

import json
import numpy as np
import logging
from tqdm import tqdm
import os

import torch
from torch import Tensor
from fakta_chat.tools.knn_searcher import KNNSearch
from fakta_chat.tools.embedder import Embedder
from transformers import AutoTokenizer, AutoModel
#from langchain.vectorstores import FAISS
#from langchain_community.vectorstores.utils import DistanceStrategy


logger = logging.getLogger(__name__)

__all__ = ['e5mistralEmbedder']
class e5mistralEmbedder(Embedder):
    def __init__(self, path_to_embedding_model: str):
        self.name = "e5-mistral-7b-instruct"
        self.max_length = 4096
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_embedding_model)
        self.model = AutoModel.from_pretrained(path_to_embedding_model)
        
    def __call__(self, texts: list[str]) -> np.array:
        return self.embed_documents(texts)
        
    #faiss expects a function called embed_documents
    def embed_documents(self, texts: list[str]) -> np.array:
        return [emb for emb in self.encode(texts)[0].numpy()]
    
    #TODO: return is always a list of one tensor with shape (1, 4096) - how to deal chat history
    def encode(self, texts: list[str]) -> list[Tensor]:
        embeddings = []
        
        for doc in texts:
            # Tokenize the document
            inputs = self.tokenizer(doc, max_length=self.max_length, return_tensors='pt', padding=True, truncation=True)
            
            # Generate the embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        return embeddings
    
    def encode_query(self, query: str, task: str = 'Given a search query, retrieve relevant passages that answer the query'):
        query = self.get_detailed_instruct(task, query)
        return self.encode([query])
    
    def last_token_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    

def index_paragraph_docs():
    path = '/data/faktalink/e5_mistral_instruct_embeddings_faiss_index'
    
    logger.info('Loading documents from Faktalink index file')
    path_to_index_file = '/data/faktalink/solr_index/index_extract_2023.json'
    with open(path_to_index_file, 'r') as file:
        data = json.load(file)
    
    e5_embedder = e5mistralEmbedder()
    db = None
    
    texts = []
    embeddings = []
    labels = []
    
    logger.info('Embedding documents and indexing them into a FAISS db with KNNSearch.')
    i = 0
    for doc in tqdm(data):
        for paragraph in doc:
            text = text = paragraph['subheadline'] + "\n" + " ".join(paragraph['sentences'])
            labels += [paragraph['id']]
            #labels += [{"id": paragraph['id'], 
            #            "link": paragraph['article_link']}]
            embeddings += [e5_embedder.embed_documents([text])]
        i += 1
        if i%10==0:
            print(f'Save FAISS db locally to {path} with {i}')
            logger.info(f'Save FAISS db locally to {path} with {i}')
            db = KNNSearch.build(np.array(embeddings), np.array(labels)) 
            db.save(index_path=path + "/embeddings", labels_path=path + "/labels")
        db = KNNSearch.build(np.array(embeddings), np.array(labels))    
    
    logger.info(f'Save FAISS db locally to {path}')
    db.save(index_path=path + "/embeddings", labels_path=path + "/labels")

if __name__ == '__main__':
    index_paragraph_docs()