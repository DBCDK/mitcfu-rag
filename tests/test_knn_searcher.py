#!/usr/bin/env python
import unittest
import numpy as np

from mitcfu_rag.tools.knn_searcher import KNNSearch


class TestKNNSearcher(unittest.TestCase):
    def test_build_raises_if_embeddings_array_is_less_than_2d(self):
        with self.assertRaises(ValueError):
            embeddings = np.random.rand(1, 2).astype("f")
            labels = np.array(["lbl_1", "lbl_2"])
            KNNSearch.build(embeddings, labels)

    def test_build_raises_if_embeddings_array_is_more_than_2d(self):
        with self.assertRaises(ValueError):
            embeddings = np.random.rand(3, 2).astype("f")
            labels = np.array(["lbl_1", "lbl_2"])
            KNNSearch.build(embeddings, labels)

    def test_build_raises_on_label_length_embedding_length_mismatch(self):
        with self.assertRaises(ValueError):
            embeddings = np.random.rand(2, 5).astype("f")
            labels = np.array(["lbl_1"])
            KNNSearch.build(embeddings, labels)

    def test_build_raises_if_embedding_dtype_is_not_float32(self):
        with self.assertRaises(TypeError):
            embeddings = np.random.rand(2, 5)
            labels = np.array(["lbl_1", "lbl_2"])
            KNNSearch.build(embeddings, labels)

    def test_search(self):
        searcher = self.__get_searcher()
        q = np.random.rand(1, 5).astype("f")

        actual = searcher.sync_search(q)
        actual_labels = [lbl for lbl, _ in actual]
        actual_similarities = np.array([sim for _, sim in actual])

        expected_labels = ["lbl_1", "lbl_2"]
        expected_similarities = np.array([0.8650935292243958, 0.6359905004501343])
        self.assertEqual(actual_labels, expected_labels)
        np.testing.assert_almost_equal(actual_similarities, expected_similarities)

    def test_search_with_k(self):
        searcher = self.__get_searcher()
        q = np.random.rand(1, 5).astype("f")

        actual = searcher.sync_search(q, k=1)
        actual_labels = [lbl for lbl, _ in actual]
        actual_similarities = np.array([sim for _, sim in actual])

        expected_labels = ["lbl_1"]
        expected_similarities = np.array([0.8650935292243958])
        self.assertEqual(actual_labels, expected_labels)
        np.testing.assert_almost_equal(actual_similarities, expected_similarities)

    def test_search_raises_if_embedding_dtype_is_not_float32(self):
        with self.assertRaises(TypeError):
            searcher = self.__get_searcher()
            q = np.random.rand(1, 5)
            searcher.sync_search(q)

    def test_search_raises_if_embedding_dimension_and_index_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            searcher = self.__get_searcher()
            q = np.random.rand(1, 7).astype("f")
            searcher.sync_search(q)

    def test_label_embedding(self):
        searcher = self.__get_searcher()
        actual = searcher.label_embeddings("lbl_2")
        expected = np.array(
            [[0.1181907, 0.23840763, 0.44230762, 0.50785077, 0.68966967]]
        )
        np.testing.assert_almost_equal(actual, expected)

    def test_label_similarity(self):
        searcher = self.__get_searcher()
        q = np.random.rand(1, 5).astype("f")

        actual = searcher.label_similarity(q, "lbl_2")
        expected = np.array([0.63599056])
        np.testing.assert_almost_equal(actual, expected)

    def test_update_adds_new_embeddings_and_labels(self):
        searcher = self.__get_searcher()
        old_ntotal = searcher.index.ntotal
        old_num_labels = len(searcher.labels)

        new_embeddings = np.random.rand(1, 5).astype("f")
        new_labels = np.array(["lbl_3"])

        searcher.update(new_embeddings, new_labels)

        # index has grown
        self.assertEqual(searcher.index.ntotal, old_ntotal + 1)
        # labels array has grown
        self.assertEqual(len(searcher.labels), old_num_labels + 1)

        labels_list = searcher.labels.tolist()
        self.assertIn("lbl_3", labels_list)

        # label2index mapping contains the new label with the expected index
        new_label_index = old_num_labels  # zero-based, appended at the end
        self.assertIn("lbl_3", searcher.label2index)
        self.assertIn(new_label_index, searcher.label2index["lbl_3"])

    def test_update_raises_if_embedding_dtype_is_not_float32(self):
        searcher = self.__get_searcher()
        embeddings = np.random.rand(1, 5)  # float64 by default
        labels = np.array(["lbl_3"])

        with self.assertRaises(TypeError):
            searcher.update(embeddings, labels)

    def test_update_raises_if_embedding_dimension_mismatch(self):
        searcher = self.__get_searcher()
        # index is built on dimension 5, so dim 7 should fail
        embeddings = np.random.rand(1, 7).astype("f")
        labels = np.array(["lbl_3"])

        with self.assertRaises(ValueError):
            searcher.update(embeddings, labels)

    def test_delete_removes_labels_and_updates_index_and_mappings(self):
        searcher = self.__get_searcher()
        labels_before = searcher.labels.tolist()
        self.assertIn("lbl_1", labels_before)

        # delete a label
        searcher.delete(["lbl_1"])

        labels_after = searcher.labels.tolist()
        self.assertNotIn("lbl_1", labels_after)
        self.assertNotIn("lbl_1", searcher.label2index)

        # label_embeddings should now fail for deleted label
        with self.assertRaises(KeyError):
            searcher.label_embeddings("lbl_1")

        # Sanity check: remaining label still works
        remaining = searcher.label_embeddings("lbl_2")
        self.assertEqual(remaining.shape[0], 1)

    def test_delete_all_labels_resets_index(self):
        searcher = self.__get_searcher()
        # delete all known labels
        searcher.delete(["lbl_1", "lbl_2"])

        # index should be empty
        self.assertEqual(searcher.index.ntotal, 0)
        # labels array should be empty
        self.assertEqual(len(searcher.labels), 0)
        # mapping should be empty
        self.assertEqual(searcher.label2index, {})

    def __get_searcher(self):
        np.random.seed(1)
        embeddings = np.random.rand(2, 5).astype("f")
        labels = np.array(["lbl_1", "lbl_2"])
        searcher = KNNSearch.build(embeddings, labels)
        return searcher
