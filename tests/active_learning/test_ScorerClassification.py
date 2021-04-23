import unittest
import numpy as np

from lightly.active_learning.scorers.classification import ScorerClassification


class TestScorerClassification(unittest.TestCase):

    def test_score_calculation(self):
        n_samples = 10000
        n_classes = 10
        np.random.seed(42)
        predictions = np.random.rand(n_samples, n_classes)
        predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
        model_output = predictions_normalized
        scorer = ScorerClassification(model_output)
        scores = scorer.calculate_scores()

        self.assertEqual(set(scores.keys()), ScorerClassification.get_score_names())

        for score_name, score in scores.items():
            self.assertEqual(score.shape, (n_samples,))
            self.assertTrue(all(score >= 0))
            self.assertTrue(all(score <= 1))


