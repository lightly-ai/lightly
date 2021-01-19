from typing import *

import numpy as np
from scipy.stats import entropy

from lightly.active_learning.scorers.scorer import Scorer


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
        uncertainties = np.array([entropy(class_probabilities) for class_probabilities in self.model_output])
        return uncertainties
