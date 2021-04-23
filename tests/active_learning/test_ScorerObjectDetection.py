import unittest

import numpy as np

from lightly.active_learning.scorers.classification import _entropy
from lightly.active_learning.utils.bounding_box import BoundingBox
from lightly.active_learning.utils.object_detection_output import ObjectDetectionOutput
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
                ],
                'labels': [
                    0,
                    1,
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
                ],
                'labels': [0]
            },
            {
                'boxes': [],
                'object_probabilities': [],
                'class_probabilities': [],
                'labels': []
            }
        ]


    def test_object_detection_scorer(self):

        # convert bounding boxes
        W, H = 128, 128
        for data in self.dummy_data:
            for i, box in enumerate(data['boxes']):
                x0 = box[0] / W
                y0 = box[1] / H
                x1 = box[2] / W
                y1 = box[3] / H
                data['boxes'][i] = BoundingBox(x0, y0, x1, y1)

        for i, data in enumerate(self.dummy_data):
            self.dummy_data[i] = ObjectDetectionOutput(
                data['boxes'],
                data['object_probabilities'],
                data['class_probabilities'],
            )

        scorer = ScorerObjectDetection(self.dummy_data)
        scores = scorer.calculate_scores()

        self.assertTrue(set(scores.keys()), set(ScorerObjectDetection.score_names()))

        res = scores['object_frequency']
        self.assertEqual(len(res), len(self.dummy_data))
        self.assertListEqual(res.tolist(), [1.0, 0.95, 0.9])

        res = scores['objectness_least_confidence']
        self.assertEqual(len(res), len(self.dummy_data))
        self.assertListEqual(res.tolist(), [0.5514945, 0.9488, 0.])

        for score_name, score in scores.items():
            if "classification" in score_name:
                self.assertEqual(len(res), len(self.dummy_data))
            if score_name == "classification_uncertainty_least_confidence":
                self.assertListEqual(list(score), [max(1 - 0.7, 1 - 0.5)/(1 - 1/3), (1 - 0.5)/(1 - 1/3), 0])
            elif score_name == "classification_uncertainty_margin":
                self.assertListEqual(list(score), [max(1 - (0.7 - 0.2), 1 - (0.5 - 0.4)), 1 - (0.5 - 0.41), 0])
            elif score_name == "classification_uncertainty_entropy":
                entropies_0 = _entropy(np.array(self.dummy_data[0].class_probabilities))/np.log2(3)
                entropies_1 = _entropy(np.array(self.dummy_data[1].class_probabilities))/np.log2(3)
                score_target = [float(max(entropies_0)), float(max(entropies_1)), 0]
                for val1, val2 in zip(score, score_target):
                    self.assertAlmostEqual(val1, val2, places=8)


    def test_object_detection_scorer_config(self):

        # convert bounding boxes
        W, H = 128, 128
        for data in self.dummy_data:
            for i, box in enumerate(data['boxes']):
                x0 = box[0] / W
                y0 = box[1] / H
                x1 = box[2] / W
                y1 = box[3] / H
                data['boxes'][i] = BoundingBox(x0, y0, x1, y1)

        for i, data in enumerate(self.dummy_data):
            self.dummy_data[i] = ObjectDetectionOutput(
                data['boxes'],
                data['object_probabilities'],
                data['class_probabilities'],
            )

        # check for default config
        scorer = ScorerObjectDetection(self.dummy_data)
        scores = scorer.calculate_scores()
        expected_default_config = {
            'frequency_penalty': 0.25,
            'min_score': 0.9
        }
        self.assertDictEqual(scorer.config, expected_default_config)

        # check for config override
        new_config = {
            'frequency_penalty': 0.55,
            'min_score': 0.6
        }
        scorer = ScorerObjectDetection(self.dummy_data, config=new_config)
        scores = scorer.calculate_scores()
        self.assertDictEqual(scorer.config, new_config)

        # check for invalid key passed
        new_config = {
            'frequenci_penalty': 0.55,
            'minimum_score': 0.6
        }
        with self.assertRaises(KeyError):
            scorer = ScorerObjectDetection(self.dummy_data, config=new_config)

        # check for wrong value passed
        new_config = {
            'frequency_penalty': 'test',
            'min_score': 1.6
        }
        with self.assertRaises(ValueError):
            scorer = ScorerObjectDetection(self.dummy_data, config=new_config)
