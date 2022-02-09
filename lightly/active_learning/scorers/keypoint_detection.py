from typing import List, Dict

import numpy as np

from lightly.active_learning.scorers import Scorer
from lightly.active_learning.utils.keypoint import Keypoint


def _least_confidence(model_output: List[List[Keypoint]]) -> np.ndarray:
    """Score which prefers samples with low confidence score.
    
    The confidence score per image is the mean confidence
    score of its keypoints.

    Args:
        model_output:
            Predictions of the model of length N.

    Returns:
        Numpy array of length N with the computed scores.

    """
    scores = []
    for keypoints in model_output:
        if len(keypoints) > 0:
            score = 1. - np.mean([keypoint.confidence for keypoint in keypoints])
        else:
            # set the score to 0 if there was no keypoint detected
            score = 0.
        scores.append(score)
    return np.asarray(scores)


class ScorerKeypointDetection(Scorer):
    """Class to compute active learning scores from the model_output of a keypoint detection task.

    Currently supports the following scorers:

        `least_confidence`:
            This scorer uses model predictions to focus more on images which
            have a low confidence of the keypoints. Use this scorer if you want scenes
            where the model is unsure about the locations the keypoints

    Attributes:
        model_output:
            List of keypoints predicted by the model for several images.
            The outer list is over all images.
            Each element of the outer list is a list of the keypoints detected
            in that image.


    Examples:
        >>> # typical model output from detectron
        >>> # see https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format
        >>> predictions = [{'pred_kepoints': np.ndarray([[123. , 456. 0.5]
        >>>     'boxes': [[0.1, 0.2, 0.3, 0.4]],
        >>>     'object_probabilities': [0.1024],
        >>>     'class_probabilities': [[0.5, 0.41, 0.09]]
        >>> }]
        >>>
        >>> # generate detection outputs
        >>> model_output = []
        >>> for prediction in predictions:
        >>>     # convert each box to a BoundingBox object
        >>>     boxes = []
        >>>     for box in prediction['boxes']:
        >>>         x0, x1 = box[0], box[2]
        >>>         y0, y1 = box[1], box[3]
        >>>         boxes.append(BoundingBox(x0, y0, x1, y1))
        >>>     # create detection outputs
        >>>     output = ObjectDetectionOutput(
        >>>         boxes,
        >>>         prediction['object_probabilities'],
        >>>         prediction['class_probabilities']
        >>>     )
        >>>     model_output.append(output)
        >>>
        >>> # create scorer from output
        >>> scorer = ScorerObjectDetection(model_output)

    """
    def __init__(self, model_output: List[List[Keypoint]]):
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
        scorer = cls(model_output=[[]])
        score_names = list(scorer.calculate_scores().keys())
        return score_names
    
