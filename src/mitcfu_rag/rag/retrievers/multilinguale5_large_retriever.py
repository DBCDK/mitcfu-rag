#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`mitcfu_rag.embedding_retriever - embedding_retriever

============
EmbeddingRetriever
============

EmbeddingRetriever retrieves relevant references based on the messages from the chat sent.

example of usage:
    from fakta_chat.embedding_retriever import EmbeddingRetriever
    d_retriever = EmbeddingRetriever()
    messages = messages = ["Hej", "Er der noget om biblioteker?"]
    refs = d_retriever.retrieve(messages)
    print(f'relevant references: {refs}')
You can also use the mitcfu-sandbox-file.ipynb to test the retriever by starting a service and querying it.
"""

import logging
from mitcfu_rag.rag.rag import Retriever, Reference
from mitcfu_rag.tools import KNNSearch
from mitcfu_rag.tools.embedder import HuggingfaceEmbedder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
import torch
import torch.nn.functional as F
from os import listdir
import json
from os.path import isfile, join

logger = logging.getLogger(__name__)


class EmbeddingRetriever(Retriever):
    def __init__(self):
        self.device = "cpu"
        self.model = AutoModel.from_pretrained(
            "/data/mitCFU-models/multilingual-e5-large/", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/data/mitCFU-models/multilingual-e5-large/", device_map="auto"
        )
        self.model.to(self.device)
        self.cross_sentence_model = AutoModelForSequenceClassification.from_pretrained(
            "/data/mitCFU-models/ms-marco-MiniLM-L-6-v2"
        )
        self.cross_sentence_tokenizer = AutoTokenizer.from_pretrained(
            "/data/mitCFU-models/ms-marco-MiniLM-L-6-v2"
        )
        # here, the searcher is loaded by specifying the path to the embeddings (index) and the labels.
        self.searcher = KNNSearch.load(
            "/data/mitcfu-rag/e5_mistral_instruct_embeddings_faiss_index/embeddings",
            "/data/mitcfu-rag/e5_mistral_instruct_embeddings_faiss_index/labels.npy",
        )
        self.all_articles = self.initiate_articles()

    def initiate_articles(self, article_folder):
        #this should also be changed, but for now this script is not used. We should probably also make the model and validator customizable
        article_folder = "/data/mitcfu-rag/test1000-jeds" 
        onlyfiles = [f for f in listdir(article_folder) if isfile(join(article_folder, f))]
        all_articles = []

        for file in onlyfiles:
            if ".json" in file and file != "index.json":
                with open(article_folder + "/" + file, "r") as f:
                    article = json.load(f)
                    all_articles.append(article)

        all_article_texts = []
        for i, article in enumerate(all_articles):
            if article.get("text"):
                for headline, text in article["text"].items():
                    if text:
                        for t in text:
                            if len(t) > 150:
                                reference = Reference(
                                    id="",
                                    article_headline=headline.strip(),
                                    #article_link=article["metadata"]["@graph"][0]["mainEntityOfPage"],
                                    text=f"{headline.strip()}: {t.strip()}",
                                )
                                all_article_texts.append(reference)

        return all_article_texts

    def retrieve(self, messages: list[str]):
        # query = f"query: {' '.join([message['content'] for message in messages if message['role'] == 'user'])}"
        query = f"query: {messages[-1]['content']}"
        # print("Query: ", query)
        return self.get_docs(query)

    # https://huggingface.co/intfloat/multilingual-e5-large
    def get_docs(self, query: str, limit: int = 3):
        batch_dict = self.tokenizer(query, max_length=512, padding=True, truncation=True, return_tensors="pt")
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embbeded_query = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy().astype(np.float32)
        hits = self.searcher.search(embbeded_query, limit)
        indexes, scores = zip(*hits)
        # references = self.filter_by_score(query, [self.all_articles[int(i)] for i in indexes])
        return list(scores), [self.all_articles[int(i)] for i in indexes][:limit]

    def filter_by_score(self, query, references, threshold=0.5):
        scores = self.cross_scores([ref.text for ref in references], query)
        return [ref for ref in references if scores[ref.text] > threshold]

    def cross_scores(self, sentences, query):
        # print("\n\nNum sentences: ", len(sentences))
        # print("\n\nSentences: ", sentences)
        features = self.cross_sentence_tokenizer(
            [query for i in range(len(sentences))], sentences, padding=True, truncation=True, return_tensors="pt"
        )
        self.cross_sentence_model.eval()
        with torch.no_grad():
            scores = self.cross_sentence_model(**features).logits.flatten()

        # Get indices of the top sentences sorted by cosine similarity
        top_indices = np.argsort(-scores)
        # print("Num top indices: ", len(top_indices))

        # Collect the top sentences and their respective cosine scores
        top_sentences_with_scores = {sentences[i]: float(scores[i]) for i in top_indices}

        return top_sentences_with_scores


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
