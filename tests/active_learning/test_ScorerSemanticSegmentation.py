import unittest

import numpy as np

import lightly
from lightly.active_learning.scorers import ScorerSemanticSegmentation
from lightly.active_learning.scorers import ScorerClassification

class TestScorerSemanticSegmentation(unittest.TestCase):

    def setUp(self):

        self.N = 100
        self.W, self.H, self.C = 32, 32, 10

        # the following data should always pass
        self.dummy_data = np.random.randn(self.N * self.W * self.H, self.C)
        self.dummy_data /= np.sum(self.dummy_data, axis=-1)[:, None]
        self.dummy_data = self.dummy_data.reshape(self.N, self.W, self.H, self.C)

        self.dummy_data_width_1 = np.random.randn(self.N * self.H, self.C)
        self.dummy_data_width_1 /= np.sum(self.dummy_data_width_1, axis=-1)[:, None]
        self.dummy_data_width_1 = self.dummy_data_width_1.reshape(self.N, 1, self.H, self.C)

        self.dummy_data_height_1 = np.random.randn(self.N * self.W, self.C)
        self.dummy_data_height_1 /= np.sum(self.dummy_data_height_1, axis=-1)[:, None]
        self.dummy_data_height_1 = self.dummy_data_height_1.reshape(self.N, self.W, 1, self.C)

        self.dummy_data_width_height_1 = np.random.randn(self.N, self.C)
        self.dummy_data_width_height_1 /= np.sum(self.dummy_data_width_height_1, axis=-1)[:, None]
        self.dummy_data_width_height_1 = self.dummy_data_width_height_1.reshape(self.N, 1, 1, self.C)

        self.dummy_data_classes_1 = np.random.randn(self.N * self.W * self.H, 1)
        self.dummy_data_classes_1 = self.dummy_data_classes_1.reshape(self.N, self.W, self.H, 1)

        # the following data should always fail
        self.dummy_data_valerr = np.random.randn(self.N, self.C)


    def test_scorer_default_case(self):
        
        scorer = ScorerSemanticSegmentation(self.dummy_data)
        scores = scorer.calculate_scores()

        for score_name, score_array in scores.items():
            self.assertTrue(isinstance(score_array, np.ndarray))
            self.assertEqual(score_array.shape, (self.N, ))

    def test_scorer_width_1_case(self):
        
        scorer = ScorerSemanticSegmentation(self.dummy_data_width_1)
        scores = scorer.calculate_scores()

        for score_name, score_array in scores.items():
            self.assertTrue(isinstance(score_array, np.ndarray))
            self.assertEqual(score_array.shape, (self.N, ))

    def test_scorer_height_1_case(self):
        
        scorer = ScorerSemanticSegmentation(self.dummy_data_height_1)
        scores = scorer.calculate_scores()

        for score_name, score_array in scores.items():
            self.assertTrue(isinstance(score_array, np.ndarray))
            self.assertEqual(score_array.shape, (self.N, ))

    def test_scorer_width_height_1_case(self):
        
        scorer = ScorerSemanticSegmentation(self.dummy_data_width_height_1)
        scores = scorer.calculate_scores()

        for score_name, score_array in scores.items():
            self.assertTrue(isinstance(score_array, np.ndarray))
            self.assertEqual(score_array.shape, (self.N, ))

    def test_scorer_classes_1_case(self):
        
        scorer = ScorerSemanticSegmentation(self.dummy_data_classes_1)
        scores = scorer.calculate_scores()

        for score_name, score_array in scores.items():
            self.assertTrue(isinstance(score_array, np.ndarray))
            self.assertEqual(score_array.shape, (self.N, ))        

    def test_wrong_input_shape(self):

        scorer = ScorerSemanticSegmentation(self.dummy_data_valerr)

        with self.assertRaises(ValueError):
            scorer.calculate_scores()