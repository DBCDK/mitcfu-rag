"""Select the document/query embedding backend explicitly.

API is the default (through GlyphGate). We do not silently fall back to a local model when it fails,
because that could mix incompatible vectors in the persisted FAISS index.
"""

from __future__ import annotations

import os

from mitcfu_rag.tools.embedder import Embedder, OpenAIEmbedder

DEFAULT_MODEL = "intfloat/multilingual-e5-large-instruct"


def create_embedder(
    *,
    backend: str | None = None,
    model: str | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    local_model_path: str | None = None,
    expected_dimensions: int | None = 1024,
) -> Embedder:
    """Instantiate an embedder using arguments or EMBEDDING_* environment values.

    EMBEDDING_BACKEND: api | local (defaults to api)
    EMBEDDING_MODEL: model ID for Glyph Gate
    EMBEDDING_API_URL: Glyph Gate or other LLM API (with /v1 or /v1/embeddings)
    EMBEDDING_API_KEY: Bearer token (required for api)
    LOCAL_EMBEDDING_MODEL_PATH: path to downloaded HF model (required for local)
    """
    selected = (backend if backend is not None else os.getenv("EMBEDDING_BACKEND", "api")).strip().lower()
    name = model if model is not None else os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)

    if selected == "api":
        return OpenAIEmbedder(
            base_url=api_url if api_url is not None else os.getenv("EMBEDDING_API_URL", ""),
            api_key=api_key if api_key is not None else os.getenv("EMBEDDING_API_KEY", ""),
            model=name,
            expected_dimensions=expected_dimensions,
        )

    if selected == "local":
        model_path = local_model_path if local_model_path is not None else os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "")
        if not model_path:
            raise ValueError("LOCAL_EMBEDDING_MODEL_PATH is required when EMBEDDING_BACKEND=local")
        # Deliberately lazy: importing the local embedder loads PyTorch/transformers.
        from mitcfu_rag.rag.retrievers.indexes.multilinguale5 import e5multilingualEmbedder

        return e5multilingualEmbedder(model_path)

    raise ValueError(f"Invalid EMBEDDING_BACKEND={selected!r}; choose 'api' or 'local'")
