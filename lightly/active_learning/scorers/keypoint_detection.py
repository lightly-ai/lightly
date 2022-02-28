from typing import List, Dict

import numpy as np

from lightly.active_learning.scorers import Scorer
from lightly.active_learning.utils.keypoint_detection_output import \
    KeypointDetectionOutput, KeypointDetection


def _least_confidence(
        model_output: List[KeypointDetectionOutput]) -> np.ndarray:
    """Score which prefers samples with low confidence score.
    
    The confidence score per image is 1 minus the mean confidence
    score of all its keypoints.

    Args:
        model_output:
            Predictions of the model of length N.

    Returns:
        Numpy array of length N with the computed scores.

    """
    scores = []
    for keypoint_detection_output in model_output:
        confidences_this_detection = []
        for keypoint_detection in keypoint_detection_output.keypoint_detections:
            confidences = keypoint_detection.get_confidences()
            if len(confidences) > 0:
                conf = np.mean(confidences)
                confidences_this_detection.append(conf)
        if len(confidences_this_detection) > 0:
            score = 1. - np.mean(confidences_this_detection)
            scores.append(score)
    return np.asarray(scores)


class ScorerKeypointDetection(Scorer):
    """Class to compute active learning scores from the model_output of a keypoint detection task.

    Currently supports the following scorers:

        `least_confidence`:
            This scorer uses model predictions to focus more on images which
            have a low confidence of the keypoints_prediction. Use this scorer if you want scenes
            where the model is unsure about the locations the keypoints_prediction

    Attributes:
        model_output:
            List of KeypointDetectionOutput predicted by the model for several images.


    Examples:
        >>>predictions_over_images = [[{
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

    """

    def __init__(self, model_output: List[KeypointDetectionOutput]):
        super(ScorerKeypointDetection, self).__init__(model_output)

    def calculate_scores(self) -> Dict[str, np.ndarray]:
        """Calculates and returns active learning scores in a dictionary.
        """
        # add classification scores
        scores = dict()
        scores['least_confidence'] = _least_confidence(self.model_output)
        return scores

    @classmethod
    def score_names(cls) -> List[str]:
        """Returns the names of the calculated active learning scores
        """
        scorer = cls(model_output=[])
        score_names = list(scorer.calculate_scores().keys())
        return score_names
