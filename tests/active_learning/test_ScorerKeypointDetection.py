import unittest

import numpy as np

from lightly.active_learning.scorers.keypoint_detection import \
    ScorerKeypointDetection
from lightly.active_learning.utils.keypoint import Keypoint


class TestScorerKeypointDetection(unittest.TestCase):

    def setUp(self) -> None:
        self.keypoints = [
            [
                Keypoint(234, 234, 0.3),
                Keypoint(456, 234, 0.5)
            ],
            [
                Keypoint(234, 432, 0.7)
            ],
            [

            ]
        ]
        self.expected_scores_least_confidence = 1 - np.array([0.4, 0.7, 1])

    def test_scorer_calculate_scores(self):

        scorer = ScorerKeypointDetection(self.keypoints)
        scores = scorer.calculate_scores()

        scores_least_confidence = scores['least_confidence']
        np.testing.assert_equal(scores_least_confidence, self.expected_scores_least_confidence)

    def test_scorer_get_score_names(self):
        scorer_1 = ScorerKeypointDetection(self.keypoints)
        scorer_2 = ScorerKeypointDetection([[]])
        self.assertGreater(len(scorer_1.score_names()), 0)
        self.assertListEqual(scorer_1.score_names(), scorer_2.score_names())




