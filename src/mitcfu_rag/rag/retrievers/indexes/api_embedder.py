import os
import logging

import numpy as np
import requests

logger = logging.getLogger(__name__)


class APIEmbedder:
    def __init__(self, model: str):
        self.url = os.environ.get("EMBEDDING_API_URL")
        self.api_key = os.environ.get("EMBEDDING_API_KEY")

        if not self.url:
            raise ValueError("EMBEDDING_API_URL is not configured")

        if not self.api_key:
            raise ValueError("EMBEDDING_API_KEY is not configured")

        self.model = model

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        logger.info(
            "Initialized API embedder with model %s",
            self.model,
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        response = self.session.post(
            self.url,
            json={
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
            },
            timeout=(10, 120),
        )

        response.raise_for_status()

        data = response.json()["data"]

        if len(data) != len(texts):
            raise ValueError(f"Expected {len(texts)} embeddings, received {len(data)}")

        ordered = sorted(data, key=lambda item: item["index"])

        embeddings = np.asarray(
            [item["embedding"] for item in ordered],
            dtype=np.float32,
        )

        # Preserve normalization from original embedder (see multilinguale5.py)
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        embeddings = embeddings / np.maximum(norms, 1e-12)

        return embeddings
