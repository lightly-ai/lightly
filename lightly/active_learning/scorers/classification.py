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
    """A class to compute active learning scores out of the model_output (i.e. the predictions of a model) for a classification task.

    Attributes:
        model_output:
            the predictions, shape: (N, C)
            N = number of samples == len(ActiveLerningAgent.unlabelled)
                the order must be the one specified by ActiveLerningAgent.unlabelled
            C = number of classes
                model_output[n,c] is the predicted probability that sample n belongs to class c
            the sum of the predictions over the classes, i.e. np.sum(model_output, axis=1),
                must equal 1 for every row/sample
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
