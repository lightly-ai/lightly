import unittest

import numpy as np
import pytest

from lightly.active_learning.scorers import scorer_classification, scorer_detection


@pytest.mark.fast
class TestScorer(unittest.TestCase):

    def test_ClassificationScorer(self):
        no_samples = 500
        no_classes = 10
        predictions = np.random.random(size=(no_samples, no_classes))
        predictions_normalized = predictions / np.sum(predictions, axis=1)[:,np.newaxis]

        scorer = scorer_classification.ScorerClassification(model_output=predictions_normalized)

        scores = scorer._calculate_scores()
