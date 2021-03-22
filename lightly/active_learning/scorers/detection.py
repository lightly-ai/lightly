from typing import *

import numpy as np

from lightly.active_learning.scorers.scorer import Scorer
from lightly.active_learning.utils.object_detection_output import ObjectDetectionOutput


def _object_frequency(model_output: List[ObjectDetectionOutput],
                      frequency_penalty: float,
                      min_score: float) -> np.ndarray:
    """Score which prefers samples with many and diverse objects.

    Args:
        model_outputs:
            Predictions of the model.
        frequency_penalty:
            Penalty applied on multiple objects of the same category. A value
            of 0.25 would count the first object fully and every additional
            object only as 0.25.
        min_score:
            The minimum score a single sample can have
        
    Returns:
        Numpy array of length N with the computed scores

    """
    n_objs = []
    for output in model_output:
        val = 0
        objs = {}
        for label in output.labels:
            if label in objs:
                objs[label] += frequency_penalty
            else:
                objs[label] = 1
        for k, v in objs.items():
            val += v
        n_objs.append(val)

    _min = min(n_objs)
    _max = max(n_objs)
    scores = [np.interp(x, (_min, _max), (min_score, 1.0)) for x in n_objs]
    return np.asarray(scores)


def _prediction_margin(model_output: List[ObjectDetectionOutput]):
    """TODO

    """
    scores = []
    for output in model_output:
        if len(output.scores) > 0:
            # prediction margin is 1 - max(class probs), therefore the mean margin
            # is mean(1 - max(class probs)) which is 1 - mean(max(class probs))
            score = 1. - np.mean(output.scores)
        else:
            # set the score to 0 if there was no bounding box detected
            score = 0.
        scores.append(score)
    return np.asarray(scores)


class ScorerObjectDetection(Scorer):
    """Class to compute active learning scores from the model_output of an object detection task.

    Attributes:
        model_output:
            TODO
        config:
            A dictionary containing additional parameters for the scorers.

    Examples:
        >>> predictions = [{
        >>>     'boxes': [[14, 16, 52, 85]],
        >>>     'object_probabilities': [0.1024],
        >>>     'class_probabilities': [[0.5, 0.41, 0.09]]
        >>> }]
        >>> scorer = ScorerObjectDetection(predictions)
    """

    def __init__(self,
                 model_output: List[ObjectDetectionOutput],
                 config: Dict = None):
        super(ScorerObjectDetection, self).__init__(model_output)
        self.config = config

    def _calculate_scores(self) -> Dict[str, np.ndarray]:
        scores = dict()
        scores['object-frequency'] = self._get_object_frequency()
        scores['prediction-margin'] = self._get_prediction_margin()
        return scores

    def _get_object_frequency(self):
        scores = _object_frequency(self.model_output, 0.25, 0.9)
        return scores

    def _get_prediction_margin(self):
        scores = _prediction_margin(self.model_output)
        return scores

