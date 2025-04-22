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

import argparse
import json
import multiprocessing as mp
import numpy as np
import logging
from tqdm import tqdm
import os
import time

import torch
from torch import Tensor
from mitcfu_rag.tools.knn_searcher import KNNSearch
from mitcfu_rag.tools.embedder import Embedder
from transformers import AutoTokenizer, AutoModel
# from langchain.vectorstores import FAISS
# from langchain_community.vectorstores.utils import DistanceStrategy


logger = logging.getLogger(__name__)

__all__ = ["e5multilingualEmbedder"]


class e5multilingualEmbedder(Embedder):
    def __init__(self, path_to_embedding_model: str):
        self.name = "multilingual-e5-large"
        self.max_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_embedding_model)
        self.model = AutoModel.from_pretrained(path_to_embedding_model)

    def __call__(self, texts: list[str]) -> np.array:
        return self.embed_documents(texts)

    # faiss expects a function called embed_documents
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts).numpy()
        # Changed from:
        # return [emb for emb in self.encode(texts)[0].numpy()]

    # TODO: return is always a list of one tensor with shape (1, max_length) - how to deal chat history
    def encode(self, texts: list[str]) -> list[Tensor]:
        # I have changed the for loop here to a batch approach, which should help if we want to use GPU
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Checking if GPU is available and switching
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logger.debug(f"Switching to device: {device}")

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self.last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])

        # Move embeddings back to CPU if GPU was used
        return embeddings.cpu()

        # embeddings = []

        # for doc in texts:
        #     # Tokenize the document
        #     inputs = self.tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)

        #     # Temporary debugging statement
        #     if inputs["input_ids"].shape[1] > self.max_length:
        #         logger.warning(f"Token length is {inputs['input_ids'].shape[1]} > max_length")

        #     # Generate the embeddings
        #     with torch.no_grad():
        #         outputs = self.model(**inputs)
        #         embeddings = self.last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        # return embeddings

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Index documents with FAISS and multilingual-e5-large from Hugging Face"
    )
    parser.add_argument(
        "--path_to_db",
        type=str,
        required=True,
        help="The path where FAISS index and labels will be stored (i.e. the vector database)",
    )
    parser.add_argument(
        "--path_to_folder",
        type=str,
        required=True,
        help="The path to the files that you want to create a vector database from",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=False,
        help="The batch size for the embedding process. If not specified, it will be set to the length of the data.",
    )
    parser.add_argument(
        "--create_new_index_extract", action="store_true", help="Create a new index extract from the folder"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Set logging level to DEBUG")
    return parser.parse_args()


# Alternative function definitions:
def index_paragraph_docs_GPU_batches(path: str, path_to_folder: str, batch_size=False, create_new_index_extract=False):
    if create_new_index_extract is True:
        logger.info(f"Creating new index extract at {path} using files from {path_to_folder}")
        onlyfiles = [f for f in os.listdir(path_to_folder) if os.path.isfile(os.path.join(path_to_folder, f))]

        # creating a list of dictionaries, by loading the (json) files in the folder
        json_files = []
        for file in tqdm(onlyfiles):
            with open(os.path.join(path_to_folder, file), "r") as f:
                data = json.load(f)
                json_files.append(data)
        # save data to a single file
        with open(path + "/test5000_index_extract_14_04_2025.json", "w") as f:
            json.dump(json_files, f)

    # load data from the file
    path_to_index_file = path + "/test5000_index_extract_14_04_2025.json"
    logger.info(f"Loading documents from {path_to_index_file}")
    # path_to_index_file = '/data/faktalink/solr_index/index_extract_2023.json'
    with open(path_to_index_file, "r") as file:
        data = json.load(file)

    e5_embedder = e5multilingualEmbedder("/data/mitCFU-models/multilingual-e5-large/")
    db = None

    # texts = []
    # if batch size was not specified, set it to the length of the data
    if batch_size is False:
        batch_size = len(data)
        logger.info(f"Batch size not specified, setting it to {batch_size}")
    else:
        logger.info(f"Batch size specified, setting it to {batch_size}")
        batch_size = int(batch_size)

    embeddings = []
    labels = []
    abstracts_to_embed_batch = []
    labels_for_batch = []
    logger.info("Embedding documents with batch approach and indexing them into a FAISS db with KNNSearch.")

    for doc in tqdm(data):
        for id in doc:
            # OBS: The abstract could be a list of strings, here we only use the first string. DETTE SKAL FIKSES:
            text = doc[str(id)].get("abstract")[0]
            if text == []:
                continue
            # before appending, check if the text is non-string type
            if not isinstance(text, str):
                logger.debug(f"Abstract is not a string: {text} in doc {doc} with id {id}")
                continue
            abstracts_to_embed_batch.append(text)
            labels_for_batch.append(str(id))

            # Check if the batch size is reached, so we can start embedding the current batch. Otherwise, we continue filling the batch
            if len(abstracts_to_embed_batch) >= batch_size:
                embeddings_for_this_batch = e5_embedder.embed_documents(abstracts_to_embed_batch)
                embeddings.extend(embeddings_for_this_batch)
                labels.extend(labels_for_batch)

                # Reset the batch for both embeddings and labels
                abstracts_to_embed_batch.clear()
                labels_for_batch.clear()

    # Check if there are any remaining embeddings in the (final) batch
    if abstracts_to_embed_batch:
        embeddings_for_this_batch = e5_embedder.embed_documents(abstracts_to_embed_batch)
        embeddings.extend(embeddings_for_this_batch)
        labels.extend(labels_for_batch)

    # Save the FAISS db locally
    logger.info(f"Saving FAISS db locally to {path}")
    db = KNNSearch.build(np.array(embeddings), np.array(labels))
    db.save(index_path=path + "/embeddings", labels_path=path + "/labels")
    logger.info(f"FAISS db saved locally to {path}")

def main():
    start = time.time()
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # index_paragraph_docs(path=args.path_to_db, path_to_folder=args.path_to_folder)
    index_paragraph_docs_GPU_batches(
        path=args.path_to_db,
        path_to_folder=args.path_to_folder,
        batch_size=args.batch_size,
        create_new_index_extract=args.create_new_index_extract,
    )

    end = time.time()
    print(f"Time taken: {round(end - start, 4)} seconds.")

if __name__ == "__main__":
    main()

"""
Example usage:
While standing in the directory with the script, run (the Python environment needs to have faiss installed):
python3 multilinguale5_instruct.py --path_to_db /data/mitcfu-rag/e5_mistral_instruct_embeddings_faiss_index_abstractover10_08_04_2025 --path_to_folder /data/mitcfu-rag/10plus-abstract-77295-jeds
This will start the indexing of the documents in the folder "--path_to_folder" and save the FAISS index and labels in the path specified after "--path_to_db".
"""


# Old function definition (without batches and GPU):
def index_paragraph_docs(path: str, path_to_folder: str):
    # path = "/data/mitcfu-rag/e5_mistral_instruct_embeddings_faiss_index_abstractover10_08_04_2025"

    logger.info(f"Loading documents from {path_to_folder}")

    # create path_to_index_file by looping through the files in the directory /data/mitcfu-rag/test1000-jeds and check if file is json
    # path_to_folder = "/data/mitcfu-rag/10plus-abstract-77295-jeds"
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

    e5_embedder = e5multilingualEmbedder("/data/faktalink_models/intfloat/multilingual-e5-large/")
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
