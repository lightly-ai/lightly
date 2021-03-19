import unittest

from lightly.active_learning.scorers.detection import ScorerObjectDetection


class TestScorerObjectDetection(unittest.TestCase):

    def setUp(self):
        self.dummy_data = [
            {
                'boxes': [
                    [14, 16, 52, 85],
                    [58, 23, 124, 49]
                ],
                'object_probabilities': [
                    0.57573,
                    0.988
                ],
                'class_probabilities': [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.5, 0.1]
                ]
            },
            {
                'boxes': [
                    [14, 16, 52, 85],
                ],
                'object_probabilities': [
                    0.1024,
                ],
                'class_probabilities': [
                    [0.5, 0.41, 0.09],
                ]
            },
            {
                'boxes': [],
                'object_probabilities': [],
                'class_probabilities': []
            }
        ]

    def test_object_frequency_scorer_works(self):
        scorer = ScorerObjectDetection(self.dummy_data)
        scores = scorer._calculate_scores()
        res = scores["object-frequency"]
        self.assertEqual(len(res), len(self.dummy_data))
        self.assertListEqual(res, [1.0, 0.95, 0.9])