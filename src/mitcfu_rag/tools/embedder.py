#!/usr/bin/env python
"""
:mod:`mitcfu_rag.tools.embedder -- Embeds texts

========
Embedder
========

Abtract base class for embedders, and huggingface implementation
"""

from abc import ABC, abstractmethod
import numpy as np
from openai import OpenAI
from sentence_transformers.SentenceTransformer import SentenceTransformer


__all__ = ["Embedder", "HuggingfaceEmbedder", "OpenAIEmbedder"]


class Embedder(ABC):
    """
    Abstract base class embedder
    """

    name: str

    @abstractmethod
    def encode(self, texts: list[str]) -> np.array:
        """encode method"""
        pass


class HuggingfaceEmbedder(Embedder):
    """Huggingface embedder. Using sentencetransformer"""

    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        """
        :param model_name:
            Name of sentence transformer model to use
        """
        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.array:
        """Encodes strings"""
        return self.model.encode(texts)


class OpenAIEmbedder(Embedder):
    """Embedder backed by an OpenAI-compatible /v1/embeddings endpoint"""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, expected_dimensions: int | None = 1024):
        """
        :param base_url:
            The `.../v1` base of the endpoint. A full `.../v1/embeddings` url is accepted too
        :param model:
            Model name sent to the endpoint
        :param api_key:
            Bearer token, if the endpoint requires one
        :param expected_dimensions:
            Expected dimensionality of the embeddings
        """
        self.model = model
        self.expected_dimensions = expected_dimensions
        # openai.OpenAI wants the `.../v1` base and appends `embeddings` itself, and refuses to start without a key
        self.client = OpenAI(base_url=base_url.rstrip("/").removesuffix("/embeddings"), api_key=api_key or "unused")

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )

        if len(response.data) != len(texts):
            raise ValueError(f"Expected {len(texts)} embeddings, received {len(response.data)}.")

        embeddings = np.asarray([d.embedding for d in sorted(response.data, key=lambda d: d.index)], dtype=np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / np.maximum(norms, 1e-12)

        if self.expected_dimensions is not None and normalized_embeddings.shape[1] != self.expected_dimensions:
            raise ValueError(
                f"Expected embeddings of dimension {self.expected_dimensions}, but got {normalized_embeddings.shape[1]}"
            )

        return normalized_embeddings

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embeds documents with the encode method."""
        return self.encode(texts)
