from typing import *

import numpy as np

from lightly.active_learning.scorers.scorer import Scorer


def entropy(probs: np.ndarray, axis: int = 1):
    zeros = np.zeros_like(probs)
    log_probs = np.log2(probs, out=zeros, where=probs > 0)
    entropies = -1 * np.sum(probs * log_probs, axis=axis)
    return entropies


class ScorerClassification(Scorer):
    def __init__(self, model_output: np.ndarray):
        self.model_output = model_output

    def _calculate_scores(self) -> Dict[str, np.ndarray]:
        scores = dict()
        scores["prediction_margin"] = self._get_prediction_margin_score()
        scores["prediction_entropy"] = self._get_prediction_entropy_score()
        return scores

    def _get_prediction_margin_score(self):
        uncertainties = np.array([1 - max(class_probabilities) for class_probabilities in self.model_output])
        return uncertainties

    def _get_prediction_entropy_score(self):
        uncertainties = entropy(self.model_output, axis=1)
        return uncertainties
