#!/usr/bin/env python
import unittest
import numpy as np

from fakta_chat.tools import KNNSearch


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
        
        actual = searcher.search(q)
        actual_labels = [lbl for lbl, _ in actual]
        actual_similarities = np.array([sim for _, sim in actual])
        
        expected_labels = ['lbl_1', 'lbl_2']
        expected_similarities = np.array([0.8650935292243958, 0.6359905004501343])        
        self.assertEqual(actual_labels, expected_labels)
        np.testing.assert_almost_equal(actual_similarities, expected_similarities)
        
    def test_search_with_k(self):
        searcher = self.__get_searcher()
        q = np.random.rand(1, 5).astype("f")
        
        actual = searcher.search(q, k=1)
        actual_labels = [lbl for lbl, _ in actual]
        actual_similarities = np.array([sim for _, sim in actual])
        
        expected_labels = ['lbl_1']
        expected_similarities = np.array([0.8650935292243958])
        self.assertEqual(actual_labels, expected_labels)
        np.testing.assert_almost_equal(actual_similarities, expected_similarities)
        
    def test_search_raises_if_embedding_dtype_is_not_float32(self):
        with self.assertRaises(TypeError):
            searcher = self.__get_searcher()            
            q = np.random.rand(1, 5)
            searcher.search(q)
    
    def test_search_raises_if_embedding_dimension_and_index_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            searcher = self.__get_searcher()            
            q = np.random.rand(1, 7).astype("f")
            searcher.search(q)

    def test_label_embedding(self):
        searcher = self.__get_searcher()
        actual = searcher.label_embeddings("lbl_2")
        expected = np.array([[0.1181907,  0.23840763, 0.44230762, 0.50785077, 0.68966967]])
        np.testing.assert_almost_equal(actual, expected)

    def test_label_similarity(self):
        searcher = self.__get_searcher()
        q = np.random.rand(1, 5).astype("f")

        actual = searcher.label_similarity(q, "lbl_2")
        expected = np.array([0.63599056])
        np.testing.assert_almost_equal(actual, expected)
        
    def __get_searcher(self):
        np.random.seed(1)
        embeddings = np.random.rand(2, 5).astype("f")
        labels = np.array(["lbl_1", "lbl_2"])        
        searcher = KNNSearch.build(embeddings, labels)
        return searcher
