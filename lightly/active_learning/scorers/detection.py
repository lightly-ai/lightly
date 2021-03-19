from typing import *

import numpy as np

from lightly.active_learning.scorers.scorer import Scorer


def _object_frequency(model_output: List,
                      frequency_pentaly: float,
                      min_score: float) -> np.ndarray:
    """
    """
    n_objs = []
    for output in model_output:
        val = 0
        objs = {}
        for label_prob in output['class_probabilities']:
            label = np.argmax(label_prob)
            if label in objs:
                objs[label] += frequency_pentaly
            else:
                objs[label] = 1
        for k, v in objs.items():
            val += v
        n_objs.append(val)

    _min = min(n_objs)
    _max = max(n_objs)
    scores = [np.interp(x, (_min, _max), (min_score, 1.0)) for x in n_objs]
    return scores


class ScorerObjectDetection(Scorer):
    """Class to compute active learning scores from the model_output of a object detection task.

    Attributes:
        model_output:
            List of length N containing the predictions of an object detection
            model. For each sample we expect a dictionary containing `boxes`,
            `object_probabilities` and `class_probabilities`.


    """
    def __init__(self, model_output: List, config: Dict=None):
        self.model_output = model_output
        self.config = config

    def _calculate_scores(self) -> Dict[str, np.ndarray]:
        scores = dict()
        scores["object-frequency"] = self._get_object_frequency()
        return scores

    def _get_object_frequency(self):
        scores = _object_frequency(self.model_output, 0.25, 0.9)
        return scores

