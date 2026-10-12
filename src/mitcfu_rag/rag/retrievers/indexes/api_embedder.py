import logging
from abc import ABC, abstractmethod

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


class Embedder(ABC):
    "Abstract base class embedder"

    name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embedding abstract method for converting texts to embeddings."""
        pass


class APIEmbedder(Embedder):
    """Embedder using an OpenAI compatible /v1/embeddings endpoint"""

    def __init__(self, base_url: str, model: str, api_key: str | None = None):
        """
        Parameters:
            base_url: The URL of the embedding API endpoint. Should be '.../v1' or '.../v1/embeddings'.
            api_key: The API key for authentication.
            model: The embedding model to use.

        Returns:
            np.ndarray: The normalized embeddings for the input texts.
        """

        self.model = model
        self.client = OpenAI(base_url=base_url.rstrip("/").removesuffix("/embeddings"), api_key=api_key or "unused")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )

        embeddings = np.asarray([d.embedding for d in sorted(response.data, key=lambda d: d.index)], dtype=np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / np.maximum(norms, 1e-12)

        return normalized_embeddings
