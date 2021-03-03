from typing import *

import numpy as np

from lightly.active_learning.scorers.scorer import Scorer


def _entropy(probs: np.ndarray, axis: int = 1) -> np.ndarray:
    """Computes the entropy of a probability matrix over one array

    Args:
        probs:
            A probability matrix of shape (N, M)
        axis:
            The axis the compute the probability over, the output does not have this axis anymore

    Exammple:
        if probs.shape = (N, C) and axis = 1 then entropies.shape = (N, )

    Returns:
        The entropy of the prediction vectors, shape: probs.shape, but without the specified axis
    """
    zeros = np.zeros_like(probs)
    log_probs = np.log2(probs, out=zeros, where=probs > 0)
    entropies = -1 * np.sum(probs * log_probs, axis=axis)
    return entropies


class ScorerClassification(Scorer):
    """Class to compute active learning scores from the model_output of a classification task.

    Attributes:
        model_output:
            Predictions of shape N x C where N is the number of unlabeled samples
            and C is the number of classes in the classification task. Must be
            normalized such that the sum over each row is 1.
            The order of the predictions must be the one specified by
            ActiveLearningAgent.unlabeled_set.

    Examples:
        >>> # example with three unlabeled samples
        >>> al_agent.unlabeled_set
        >>> > ['img0.jpg', 'img1.jpg', 'img2.jpg']
        >>> predictions = np.array(
        >>>     [
        >>>          [0.1, 0.9], # predictions for img0.jpg
        >>>          [0.3, 0.7], # predictions for img1.jpg
        >>>          [0.8, 0.2], # predictions for img2.jpg
        >>>     ] 
        >>> )
        >>> np.sum(predictions, axis=1)
        >>> > array([1., 1., 1.])
        >>> scorer = ScorerClassification(predictions)

    """
    def __init__(self, model_output: np.ndarray):
        self.model_output = model_output

    def _calculate_scores(self) -> Dict[str, np.ndarray]:
        scores = dict()
        scores["prediction-margin"] = self._get_prediction_margin_score()
        scores["prediction-entropy"] = self._get_prediction_entropy_score()
        return scores

    def _get_prediction_margin_score(self):
        uncertainties = np.array([1 - max(class_probabilities) for class_probabilities in self.model_output])
        return uncertainties

    def _get_prediction_entropy_score(self):
        uncertainties = _entropy(self.model_output, axis=1)
        return uncertainties
