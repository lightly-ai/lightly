import unittest

import numpy as np

from lightly.active_learning.scorers.keypoint_detection import \
    ScorerKeypointDetection

from lightly.active_learning.utils.keypoint_detection_output import \
    KeypointDetection, KeypointDetectionOutput


class TestScorerKeypointDetection(unittest.TestCase):

    def setUp(self) -> None:
        predictions_over_images = [[{
            'pred_keypoints': np.asarray([[123., 456., 0.1], [565., 32., 0.2]])
        }, {
            'pred_keypoints': np.asarray([[342., 432., 0.3], [43., 2., 0.4]])}
        ], [{
            'pred_keypoints': np.asarray([[23., 43., 0.5], [43., 2., 0.6]])
        }]]
        model_output = []
        for predictions_one_image in predictions_over_images:
            keypoint_detections = []
            for prediction in predictions_one_image:
                keypoints = prediction['pred_keypoints'].flatten()
                keypoint_detection = KeypointDetection(keypoints)
                keypoint_detections.append(keypoint_detection)
            output = KeypointDetectionOutput(keypoint_detections)
            model_output.append(output)

        self.model_output = model_output
        self.expected_scores_least_confidence = np.asarray([0.75, 0.45])

    def test_scorer_calculate_scores(self):

        scorer = ScorerKeypointDetection(self.model_output)
        scores = scorer.calculate_scores()

        scores_least_confidence = scores['least_confidence']
        np.testing.assert_allclose(scores_least_confidence, self.expected_scores_least_confidence)

    def test_scorer_get_score_names(self):
        scorer_1 = ScorerKeypointDetection(self.model_output)
        scorer_2 = ScorerKeypointDetection([])
        self.assertGreater(len(scorer_1.score_names()), 0)
        self.assertListEqual(scorer_1.score_names(), scorer_2.score_names())




