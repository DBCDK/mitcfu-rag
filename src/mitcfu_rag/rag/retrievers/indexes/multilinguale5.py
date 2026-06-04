#!/usr/bin/env python
"""
:mod:`mitcfu embedder med modellen multilingual-e5-large

========
e5multilingualEmbedder
========

e5multilingualEmbedder embeds JEDs with e5-mistral-7b-instruct model from Huggingface.
multilingual-e5-large ranks high on ScanEval for Danish (total: 60.7) and can encode 512 tokens.
https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/
Embeddings can be stored as FAISS db for fast retrieval.
See:
https://huggingface.co/intfloat/multilingual-e5-large
"""

import numpy as np
import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from mitcfu_rag.tools.embedder import Embedder
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

__all__ = ["e5multilingualEmbedder"]


class e5multilingualEmbedder(Embedder):
    def __init__(self, path_to_embedding_model: str):
        self.name = "multilingual-e5-large-instruct"
        self.max_length = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_embedding_model, device_map=self.device)
        self.model = AutoModel.from_pretrained(path_to_embedding_model, device_map=self.device)
        self.model.eval()

    def __call__(self, texts: list[str]) -> np.array:
        return self.embed_documents(texts)

    # faiss expects a function called embed_documents
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts).numpy()

    def encode(self, texts: list[str]) -> list[Tensor]:
        # I have changed the for loop here to a batch approach, which should help if we want to use GPU
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            # Checking if GPU is available and switching
            def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
                return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

            # Tokenize the document
            batch_dict = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"]).float()
            embedded_passage = F.normalize(embeddings, p=2, dim=1).detach().cpu()
        return embedded_passage

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def get_detailed_instruct(self, query: str) -> str:
        return f"{query}"
