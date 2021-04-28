import unittest
import numpy as np

from lightly.active_learning.scorers.classification import ScorerClassification, _entropy


class TestScorerClassification(unittest.TestCase):

    def test_score_calculation_random(self):
        n_samples = 10000
        n_classes = 10
        np.random.seed(42)
        predictions = np.random.rand(n_samples, n_classes)
        predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
        model_output = predictions_normalized
        scorer = ScorerClassification(model_output)
        scores = scorer.calculate_scores()

        self.assertEqual(set(scores.keys()), set(ScorerClassification.score_names()))

        for score_name, score in scores.items():
            self.assertEqual(score.shape, (n_samples,))
            self.assertTrue(all(score >= 0))
            self.assertTrue(all(score <= 1))

    def test_score_calculation_specific(self):
        model_output = [
            [0.7, 0.2, 0.1],
            [0.4, 0.5, 0.1]
        ]
        model_output = np.array(model_output)
        scorer = ScorerClassification(model_output)
        scores = scorer.calculate_scores()

        self.assertListEqual(list(scores["uncertainty_least_confidence"]),
                             [(1 - 0.7) / (1 - 1. / 3.), (1 - 0.5) / (1 - 1. / 3.)])
        self.assertListEqual(list(scores["uncertainty_margin"]), [1 - (0.7 - 0.2), 1 - (0.5 - 0.4)])
        for val1, val2 in zip(scores["uncertainty_entropy"], _entropy(model_output) / np.log2(3)):
            self.assertAlmostEqual(val1, val2, places=8)

    def test_scorer_classification_empty_model_output(self):
        scorer = ScorerClassification(model_output=[])
        scores = scorer.calculate_scores()
        self.assertEqual(set(scores.keys()), set(ScorerClassification.score_names()))

    def test_scorer_classification_variable_model_output_dimension(self):

        for num_samples in range(5):
            for num_classes in range(5):

                with self.subTest(msg=f"model_output.shape = ({num_samples},{num_classes})"):
                    if num_samples > 0:
                        preds = [1. / num_samples] * num_classes
                    else:
                        preds = []
                    model_output = [preds] * num_samples

                    if num_classes == 0 and num_samples > 0:
                        with self.assertRaises(ValueError):
                            scorer = ScorerClassification(model_output=model_output)
                    else:
                        scorer = ScorerClassification(model_output=model_output)
                        scores = scorer.calculate_scores()
                        self.assertEqual(set(scores.keys()), set(ScorerClassification.score_names()))
                        for score_values in scores.values():
                            self.assertEqual(len(score_values), len(model_output))

    def test_scorer_classification_variable_model_output_tensor_order(self):

        for tensor_order in range(1, 5):
            model_output = np.ndarray((3,) * tensor_order)
            with self.subTest(msg=f"model_output.shape = {model_output.shape}"):
                if tensor_order == 2 or tensor_order == 0:
                    scorer = ScorerClassification(model_output=model_output)
                    scores = scorer.calculate_scores()
                else:
                    with self.assertRaises(ValueError):
                        scorer = ScorerClassification(model_output=model_output)
