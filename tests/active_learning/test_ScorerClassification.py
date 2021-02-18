import unittest
import numpy as np

from lightly.active_learning.scorers.classification import ScorerClassification


class TestScorerClassification(unittest.TestCase):

    def test_score_calculation(self):
        n_samples = 100
        n_classes = 10
        predictions = np.random.rand(n_samples, n_classes)
        predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
        model_output = predictions_normalized
        scorer = ScorerClassification(model_output)
        scores = scorer._calculate_scores()
        scores_prediction_entropy = scores["prediction_entropy"]
        scores_prediction_margin = scores["prediction_margin"]

        assert scores_prediction_entropy.shape == (n_samples, )
        assert scores_prediction_margin.shape == (n_samples, )
        assert all(scores_prediction_entropy > 0)
        assert all(scores_prediction_margin > 0)
        assert all(scores_prediction_margin < 1)
