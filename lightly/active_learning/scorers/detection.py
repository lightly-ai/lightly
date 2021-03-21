from typing import *

import numpy as np

from lightly.active_learning.scorers.scorer import Scorer


def _object_frequency(model_output: List[Dict],
                      frequency_penalty: float,
                      min_score: float) -> np.ndarray:
    """Scorer which prefers samples with many and diverse objects

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
        for label_prob in output['class_probabilities']:
            label = np.argmax(label_prob)
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


class ScorerObjectDetection(Scorer):
    """Class to compute active learning scores from the model_output of a object detection task.

    Attributes:
        model_output:
            List of length N containing the predictions of an object detection
            model. For each sample we expect a dictionary containing `boxes`,
            `object_probabilities` and `class_probabilities`.
            `boxes` has shape [B, 4] for B found bounding boxes and the 4 values
            x1, y1, x2, y2.
            `object_probabilities` has shape [B] with a objectness value for each
            bounding box.
            `class_probabilities` has shape [B, C] for the B bounding boxes and
            C classes. The class probabilities should all sum up to 1.
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

    def __init__(self, model_output: List[Dict], config: Dict=None):
        self.model_output = model_output
        self.config = config

    def _calculate_scores(self) -> Dict[str, np.ndarray]:
        scores = dict()
        scores["object-frequency"] = self._get_object_frequency()
        return scores

    def _get_object_frequency(self):
        scores = _object_frequency(self.model_output, 0.25, 0.9)
        return scores

