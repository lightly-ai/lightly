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
        scores = scorer.calculate_scores()

        for score_name, score in scores.items():
            assert all(score >= 0)
            assert score.shape == (n_samples,)
            if "entropy" not in score_name:
                assert all(score <= 1)
