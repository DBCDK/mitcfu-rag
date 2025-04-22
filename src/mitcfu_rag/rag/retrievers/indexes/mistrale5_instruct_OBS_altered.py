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

__all__ = ["e5mistralEmbedder"]

import json
import multiprocessing as mp
import numpy as np
import logging
from tqdm import tqdm
import os

import torch
from torch import Tensor
from mitcfu_rag.tools.knn_searcher import KNNSearch
from mitcfu_rag.tools.embedder import Embedder
from transformers import AutoTokenizer, AutoModel
# from langchain.vectorstores import FAISS
# from langchain_community.vectorstores.utils import DistanceStrategy


logger = logging.getLogger(__name__)

__all__ = ["e5mistralEmbedder"]


class e5mistralEmbedder(Embedder):
    def __init__(self, path_to_embedding_model: str):
        self.name = "e5-mistral-7b-instruct"
        self.max_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_embedding_model)
        self.model = AutoModel.from_pretrained(path_to_embedding_model)

    def __call__(self, texts: list[str]) -> np.array:
        return self.embed_documents(texts)

    # faiss expects a function called embed_documents
    def embed_documents(self, texts: list[str]) -> np.array:
        return [emb for emb in self.encode(texts)[0].numpy()]

    # TODO: return is always a list of one tensor with shape (1, max_length) - how to deal chat history
    def encode(self, texts: list[str]) -> list[Tensor]:
        embeddings = []

        for doc in texts:
            # Tokenize the document
            inputs = self.tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)

            # Temporary debugging statement
            if inputs["input_ids"].shape[1] > self.max_length:
                logger.warning(f"Token length is {inputs['input_ids'].shape[1]} > max_length")

            # Generate the embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self.last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        return embeddings

    def encode_query(
        self, query: str, task: str = "Given a search query, retrieve relevant passages that answer the query"
    ):
        query = self.get_detailed_instruct(task, query)
        return self.encode([query])

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"


def index_paragraph_docs():
    path = "/data/mitcfu-rag/e5_mistral_instruct_embeddings_faiss_index_abstractover10_08_04_2025"

    logger.info("Loading documents from Faktalink index file")

    # create path_to_index_file by looping through the files in the directory /data/mitcfu-rag/test1000-jeds and check if file is json
    path_to_folder = "/data/mitcfu-rag/10plus-abstract-77295-jeds"
    onlyfiles = [f for f in os.listdir(path_to_folder) if os.path.isfile(os.path.join(path_to_folder, f))]
    # create a list of the content of the json files, with each file being a list of dictionaries
    json_files = []
    for file in tqdm(onlyfiles):
        with open(os.path.join(path_to_folder, file), "r") as f:
            data = json.load(f)
            json_files.append(data)
    # save data to a single file
    with open(path + "/index_extract_08_04_2025.json", "w") as f:
        json.dump(json_files, f)

    # load data from the file
    path_to_index_file = path + "/index_extract_08_04_2025.json"
    # path_to_index_file = '/data/faktalink/solr_index/index_extract_2023.json'
    with open(path_to_index_file, "r") as file:
        data = json.load(file)

    e5_embedder = e5mistralEmbedder("/data/faktalink_models/intfloat/multilingual-e5-large/")
    db = None

    # texts = []
    embeddings = []
    labels = []

    logger.info("Embedding documents and indexing them into a FAISS db with KNNSearch.")
    i = 0
    for doc in tqdm(data):
        for id in doc:
            text = doc[str(id)].get("abstract")
            print(f"Embedding abstract: {text}")

            # If the abstract is empty, we skip to the next iteration
            if text == []:
                continue

            labels.append(str(id))
            # labels += [{"id": paragraph['id'],
            #            "link": paragraph['article_link']}]
            embedding = e5_embedder.embed_documents([text])
            embeddings.append(embedding)
        i += 1
        if i % 10 == 0:
            print(f"Save FAISS db locally to {path} with {i}")
            logger.info(f"Save FAISS db locally to {path} with {i}")
            db = KNNSearch.build(np.array(embeddings), np.array(labels))
            db.save(index_path=path + "/embeddings", labels_path=path + "/labels")
        db = KNNSearch.build(np.array(embeddings), np.array(labels))

    logger.info(f"Save FAISS db locally to {path}")
    db.save(index_path=path + "/embeddings", labels_path=path + "/labels")


if __name__ == "__main__":
    index_paragraph_docs()
