#!/usr/bin/env python
""":mod:`mitcfu_rag.tools.knn_searcher -- knn search engine

=========
KNNSearch
=========

K-Nearest neighbour search engine using cosine similarity.
The searcher is build by providing a embedding array and associated
labels. The labels do not need to be unique (if not they can be
present multiple times in a search result). The class also implements
some helper methods and a handy load and save functionality.

NOTE: The cast to float (astype('f')) in the example below is not
      necessary when using output from sentence-transformer, as this
      already is float32.

Example of usage:

    In [1]: from mitcfu_rag.tools import KNNSearch
            import numpy as np

            np.random.seed(1)
            embeddings = np.random.rand(5, 5).astype('f')
            labels = np.array(['label_1', 'label_4', 'label_3', 'label_2', 'label_4'])
            searcher = KNNSearch.build(embeddings, labels)
            q = np.random.rand(1, 5).astype('f')


    In [2]: searcher.search(q, 4)
    Out[2]:
    [('label_4', 0.7828297019004822),
     ('label_2', 0.6757404804229736),
     ('label_4', 0.6583917737007141),
     ('label_1', 0.5397369265556335)]

    In [3]: searcher.label_embeddings('label_4')
    Out[3]:
    array([[0.1181907 , 0.23840763, 0.44230762, 0.50785077, 0.68966967],
           [0.46825364, 0.566213  , 0.18328191, 0.4048514 , 0.5124885 ]],
          dtype=float32)

    In [4]: searcher.label_similarity(q, 'label_3')
    Out[4]: array([0.40104464], dtype=float32)

"""

from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

__all__ = ["KNNSearch"]


File = str | Path


class KNNSearch:
    """
    K-Nearest neighbour search engine.

    Search engine to find k nearest embeddings using cosine
    similarity.
    """

    def __init__(self, index, labels: np.array):
        """Build index with the build method"""
        self.index = index
        self.labels = labels
        self.label2index = defaultdict(set)
        for i, label in enumerate(self.labels):
            self.label2index[label].add(i)
        self.label2index = dict(self.label2index)
        self.executor = ThreadPoolExecutor(max_workers=5)

    @classmethod
    def build(cls, embeddings: np.array, labels: np.array):
        """
        Builds KNNSearch instance from array of embeddings and array of
        labels. length of axis 0 must be the same for embeddings and
        labels. The labels don't need to be unique.

        :param embeddings:
            Embeddings to index into knn index
        :param labels:
            Labels corresponding to embeddings
        """
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embedding array must be 2D (found {len(embeddings.shape)}D)")
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(f"Incompatible lengths: embeddings={embeddings.shape[0]} vs labels={labels.shape[0]}")
        if embeddings.dtype != np.float32:
            raise TypeError(f"embeddings array must be of dtype float32 ({embeddings.dtype} found)")

        index = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        embeddings_copy = embeddings.copy()
        faiss.normalize_L2(embeddings_copy)
        index.add(embeddings_copy)
        return cls(index, labels)

    def __search(self, embedding, k):
        return self.index.search(embedding, k)

    async def async_search(self, embedding, k):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.__search, embedding, k)

    async def search(self, embedding: np.array, k: int = 10, min_similarity: float = 0.1) -> list[tuple[str, float]]:
        """
        Searches for k nearest neighbours among indexed embeddings using cosine similarity.

        returns list of labels and similarity

        :param embedding:
            Embedding to find nearest neighbours for
        :param k:
            max number of neighbours to find
        :param min_similarity:
            Minimum similarity of returned neighbours
        """
        embedding = self.__copy_and_normalize(embedding)
        similarities, indices = await self.async_search(embedding, k)
        similarities = similarities[0, :]
        indices = indices[0, :]

        result = [
            (str(self.labels[idx]), float(sim)) for sim, idx in zip(similarities, indices) if sim >= min_similarity
        ]
        return result

    def sync_search(self, embedding: np.array, k: int = 10, min_similarity: float = 0.1):
        """
        Synchronous version of search().

        Runs a nearest-neighbor search without asyncio or executors.

        :param embedding:
            Embedding to search for.
        :param k:
            Number of neighbors to retrieve.
        :param min_similarity:
            Minimum similarity required for results.
        :return:
            List of (label, similarity) tuples.
        """
        # Normalize embedding (same logic as async version)
        embedding = self.__copy_and_normalize(embedding)

        # Direct FAISS call — no async, no executor
        similarities, indices = self.__search(embedding, k)

        similarities = similarities[0, :]
        indices = indices[0, :]

        # Build result list
        result = [
            (str(self.labels[idx]), float(sim)) for sim, idx in zip(similarities, indices) if sim >= min_similarity
        ]

        return result

    def label_embeddings(self, label: str) -> np.array:
        """
        Return embedding(s) associated with label.

        :param label:
            Label to retrieve embeddings for
        """
        return np.array([self.index.reconstruct(int(i)) for i in self.label2index[label]])

    def label_similarity(self, embedding: np.array, label: str, k: int = None) -> np.array:
        """
        Find cosine similarity between supplied embedding and the
        embeddings associated with the supplied label. Returns the n
        closest similarities.

        :param embedding:
            The embedding to compare label emebddings with
        :param label:
            Label to look up embeddings for
        :paramn k:
            Return at most n similarities
        """
        label_embeddings = self.label_embeddings(label)
        embedding = self.__copy_and_normalize(embedding)
        sims = cosine_similarity(embedding, label_embeddings)
        ordered_sims = np.flip(np.sort(sims))
        if k:
            ordered_sims = ordered_sims[:k]
        return ordered_sims[0, :]

    def save(self, index_path: File, labels_path: File):
        """
        Saves knn-search to disk (two files).

        :param index_path:
            Path to save index at
        :param labels_path:
            Path to save labels at
        """
        faiss.write_index(self.index, str(index_path))
        np.save(str(labels_path), self.labels)

    @classmethod
    def load(cls, index_path: File, labels_path: File):
        """
        loads knn-search from files.

        :param index_path:
            Path to load index from
        :param labels_path:
            Path to load labels from
        """
        index = faiss.read_index(str(index_path))
        labels = np.load(str(labels_path), allow_pickle=True)
        return cls(index, labels)

    def update(self, embeddings: np.array, labels: np.array) -> None:
        """
        Append new embeddings and labels to the existing index.

        :param embeddings:
            2D array of embeddings to add (float32, same dim as existing index)
        :param labels:
            1D array of labels corresponding to the embeddings
        """
        # Allow noop
        if embeddings is None or labels is None:
            return
        if embeddings.size == 0 or labels.size == 0:
            return

        # Basic shape checks (same as in build)
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embedding array must be 2D (found {len(embeddings.shape)}D)")
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(f"Incompatible lengths: embeddings={embeddings.shape[0]} vs labels={labels.shape[0]}")
        if embeddings.dtype != np.float32:
            raise TypeError(f"embeddings array must be of dtype float32 ({embeddings.dtype} found)")
        if embeddings.shape[1] != self.index.d:
            raise ValueError(f"Incompatible embedding dimension: {embeddings.shape[1]} vs {self.index.d}")

        # Normalize and add to FAISS index
        embeddings_copy = embeddings.copy()
        faiss.normalize_L2(embeddings_copy)
        self.index.add(embeddings_copy)

        # Extend labels
        start_idx = len(self.labels)
        self.labels = np.concatenate([self.labels, labels])

        # Update label2index mapping
        for offset, label in enumerate(labels):
            idx = start_idx + offset
            if label in self.label2index:
                self.label2index[label].add(idx)
            else:
                self.label2index[label] = {idx}

    def delete(self, labels_to_delete) -> None:
        """
        Delete all embeddings (and labels) associated with the supplied labels.

        The FAISS index and internal mappings are rebuilt so that
        subsequent searches and label lookups reflect the deletions.

        :param labels_to_delete:
            Iterable of labels to delete. Any label not present is ignored.
        """
        if not labels_to_delete:
            return

        labels_to_delete = set(labels_to_delete)

        # Indices we want to keep
        keep_indices = [i for i, label in enumerate(self.labels) if label not in labels_to_delete]

        # Nothing to delete
        if len(keep_indices) == len(self.labels):
            return

        # Everything deleted: reset to empty index and mappings
        if not keep_indices:
            self.index = faiss.index_factory(self.index.d, "Flat", faiss.METRIC_INNER_PRODUCT)
            self.labels = np.empty((0,), dtype=self.labels.dtype)
            self.label2index = {}
            return

        # Reconstruct all embeddings from current index
        all_embeddings = np.vstack([self.index.reconstruct(i) for i in range(self.index.ntotal)]).astype("f")

        new_embeddings = all_embeddings[keep_indices]
        new_labels = self.labels[keep_indices]

        # Rebuild index and mappings using existing class logic
        rebuilt = self.build(new_embeddings, new_labels)
        self.index = rebuilt.index
        self.labels = rebuilt.labels
        self.label2index = rebuilt.label2index

    def delete_by_prefix(self, prefixes) -> None:
        """
        Delete all embeddings (and labels) whose label starts with ANY of the
        supplied prefixes.

        example: mitcfu-id 48984851 is to be deleted -> Deletes labels 48984851_chunk0 and 48984851_chunk1.

        :param prefixes:
            A single prefix string, or an iterable of prefix strings.
        """
        # Normalize to a tuple of strings
        if isinstance(prefixes, str):
            prefixes = (prefixes,)
        else:
            prefixes = tuple(prefixes)

        if not prefixes:
            return

        # Work on a string view of labels
        labels_str = self.labels.astype(str)
        prefixes_arr = np.array(prefixes, dtype=str)

        # Broadcasted startswith:
        # labels_str:      shape (N,)
        # prefixes_arr:    shape (P,)
        # result:          shape (N, P)
        matches = np.char.startswith(labels_str[:, None], prefixes_arr[None, :])

        # Boolean mask of labels that match ANY prefix
        mask = matches.any(axis=1)

        if not mask.any():
            return

        # Labels to delete (as strings)
        labels_to_delete = labels_str[mask]

        # Reuse existing delete() logic
        self.delete(list(labels_to_delete))

    def __copy_and_normalize(self, embedding: np.array) -> np.array:
        embedding_copy = embedding.copy()
        if len(embedding_copy.shape) == 1:
            embedding_copy = embedding_copy.reshape(1, -1)
        if embedding_copy.shape[1] != self.index.d:
            raise ValueError(f"Incompatible shapes: {embedding_copy.shape[1]} vs {self.index.d}")
        if embedding.dtype != np.float32:
            raise TypeError(f"embedding array must be of dtype float32 ({embedding.dtype} found)")
        faiss.normalize_L2(embedding_copy)
        return embedding_copy
