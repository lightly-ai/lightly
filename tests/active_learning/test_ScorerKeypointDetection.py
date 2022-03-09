import unittest

import numpy as np

from lightly.active_learning.scorers.keypoint_detection import \
    ScorerKeypointDetection

from lightly.active_learning.utils.keypoint_predictions import \
    KeypointInstancePrediction, KeypointPrediction


class TestScorerKeypointDetection(unittest.TestCase):

    def setUp(self) -> None:
        predictions_over_images = [
            [
                {
                    "keypoints": [123., 456., 0.1, 565., 32., 0.2]
                }, {
                    "keypoints": [342., 432., 0.3, 43., 2., 0.4]
                }
            ], [
                {
                    "keypoints": [23., 43., 0.5, 43., 2., 0.6]
                }
            ], [
            ]
        ]
        model_output = []
        for predictions_one_image in predictions_over_images:
            keypoint_detections = []
            for prediction in predictions_one_image:
                keypoints = prediction["keypoints"]
                keypoint_detection = KeypointInstancePrediction(keypoints)
                keypoint_detections.append(keypoint_detection)
            output = KeypointPrediction(keypoint_detections)
            model_output.append(output)

        self.model_output = model_output
        self.expected_scores_mean_uncertainty = np.asarray([0.75, 0.45, 0])

    def test_scorer_calculate_scores(self):

        scorer = ScorerKeypointDetection(self.model_output)
        scores = scorer.calculate_scores()

        scores_mean_uncertainty = scores["mean_uncertainty"]
        np.testing.assert_allclose(scores_mean_uncertainty, self.expected_scores_mean_uncertainty)

    def test_scorer_get_score_names(self):
        scorer_1 = ScorerKeypointDetection(self.model_output)
        scorer_2 = ScorerKeypointDetection([])
        self.assertGreater(len(scorer_1.score_names()), 0)
        self.assertListEqual(scorer_1.score_names(), scorer_2.score_names())

    def test_keypoint_instance_prediction_creation(self):
        with self.subTest("create correct"):
            KeypointInstancePrediction([456., 32., 0.3])
        with self.subTest("create correct with object_id"):
            KeypointInstancePrediction([456., 32., 0.3], 3)
        with self.subTest("create correct with object_id and score"):
            KeypointInstancePrediction([456., 32., 0.3], 3, 0.3)
        with self.subTest("create correct with score"):
            KeypointInstancePrediction([456., 32., 0.3], score = 0.3)
        with self.subTest("create wrong keypoints format"):
            with self.assertRaises(ValueError):
                KeypointInstancePrediction([456., 32., 0.3, 1], 3)
        with self.subTest("create confidence < 0"):
            with self.assertRaises(ValueError):
                KeypointInstancePrediction([456., 32., -0.1], 3)
        with self.subTest("create confidence > 1"):
            with self.assertRaises(ValueError):
                KeypointInstancePrediction([456., 32., 1.5], 3)
        with self.subTest("create from dict"):
            dict_ = {
                "category_id": 3,
                "keypoints": [423, 432, 0.4, 231, 655, 0.3],
                "score": -1.9
            }
            KeypointInstancePrediction.from_dict(dict_)

    def test_keypoint_prediction_creation(self):
        with self.subTest("create from KeypointInstancePrediction"):
            keypoints = [
                KeypointInstancePrediction([456., 32., 0.3]),
                KeypointInstancePrediction([456., 32., 0.3], 3, 0.3)
            ]
            KeypointPrediction(keypoints)
        with self.subTest("create from dicts"):
            dicts = [
                {
                    "category_id": 3,
                    "keypoints": [423, 432, 0.4, 231, 655, 0.3],
                    "score": -1.9
                }
            ]
            KeypointPrediction.from_dicts(dicts)

        with self.subTest("create from string"):
            json_str = """[
                {
                    "category_id": 3, 
                    "keypoints": [423, 432, 0.4, 231, 655, 0.3], 
                    "score": -1.9
                }
            ]"""
            KeypointPrediction.from_json_string(json_str)




