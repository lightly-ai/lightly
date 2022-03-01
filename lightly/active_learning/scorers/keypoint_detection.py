from typing import List, Dict

import numpy as np

from lightly.active_learning.scorers import Scorer
from lightly.active_learning.utils.keypoint_predictions import \
    KeypointPrediction


def _mean_uncertainty(
        model_output: List[KeypointPrediction]) -> np.ndarray:
    """Score which prefers samples with low confidence score.
    
    The uncertainty score per image is 1 minus the mean confidence
    score of all its keypoints.

    Args:
        model_output:
            Predictions of the model of length N.

    Returns:
        Numpy array of length N with the computed scores.

    """
    scores = []
    for keypoint_prediction in model_output:
        confidences_image = []
        for keypoint_instance_prediction in keypoint_prediction.keypoint_instance_predictions:
            confidences_instance = keypoint_instance_prediction.get_confidences()
            if len(confidences_instance) > 0:
                conf = np.mean(confidences_instance)
                confidences_image.append(conf)
        if len(confidences_image) > 0:
            score = 1. - np.mean(confidences_image)
            scores.append(score)
        else:
            scores.append(0)
    return np.asarray(scores)


class ScorerKeypointDetection(Scorer):
    """Class to compute active learning scores from the model_output of a keypoint detection task.

    Currently supports the following scorers:

        `mean_uncertainty`:
            This scorer uses model predictions to focus more on images which
            have a low mean confidence of the predicted keypoints.
            Use this scorer if you want scenes
            where the model is unsure about the locations of the keypoints.

    Attributes:
        model_output:
            List of KeypointDetectionOutput predicted by the model for several images.


    Examples:
        >>> predictions_over_images = [[{
        >>>     'keypoints': [123., 456., 0.1, 565., 32., 0.2]
        >>> }, {
        >>>     'keypoints': [432., 34., 0.1, 43., 34., 0.3]}
        >>> ], [{
        >>>     'keypoints': [123., 456., 0.1, 565., 32., 0.2])
        >>> }],
        >>> ]
        >>> model_output = []
        >>> for predictions_one_image in predictions_over_images:
        >>>     keypoint_detections = []
        >>>     for prediction in predictions_one_image:
        >>>         keypoints = prediction['keypoints']
        >>>         keypoint_detection = KeypointInstancePrediction(keypoints)
        >>>         keypoint_detections.append(keypoint_detection)
        >>>     output = KeypointPrediction(keypoint_detections)
        >>>     model_output.append(output)
        >>> scorer = ScorerKeypointDetection(model_output)
        >>> scores = scorer.calculate_scores()

    """

    def __init__(self, model_output: List[KeypointPrediction]):
        super(ScorerKeypointDetection, self).__init__(model_output)

    def calculate_scores(self) -> Dict[str, np.ndarray]:
        """Calculates and returns active learning scores in a dictionary.
        """
        # add classification scores
        scores = dict()
        scores['mean_uncertainty'] = _mean_uncertainty(self.model_output)
        return scores

    @classmethod
    def score_names(cls) -> List[str]:
        """Returns the names of the calculated active learning scores
        """
        scorer = cls(model_output=[])
        score_names = list(scorer.calculate_scores().keys())
        return score_names
