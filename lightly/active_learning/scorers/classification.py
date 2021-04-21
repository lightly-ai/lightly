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

def _margin_largest_secondlargest(probs: np.ndarray) -> np.ndarray:
    """Computes the margin of a probability matrix

        Args:
            probs:
                A probability matrix of shape (N, M)

        Exammple:
            if probs.shape = (N, C) then margins.shape = (N, )

        Returns:
            The margin of the prediction vectors
        """
    sorted_probs = np.partition(probs, -2, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    return margins


class ScorerClassification(Scorer):
    """Class to compute active learning scores from the model_output of a classification task.

    Currently supports the following scorers:

        `prediction-margin`:
            This scorer uses the margin between 1.0 and the highest confidence
            prediction. Use this scorer to select images where the model is
            insecure.

        `prediction-entropy`:
            This scorer computes the entropy of the prediction. All
            confidences are considered to compute the entropy of a sample.

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
        super(ScorerClassification, self).__init__(model_output)

    def calculate_scores(self) -> Dict[str, np.ndarray]:
        """Calculates and returns the active learning scores.

        Returns:
            A dictionary mapping from the score name (as string)
            to the scores (as a single-dimensional numpy array).
        """

        scores_with_names = [
            self._get_scores_uncertainty_least_confidence(),
            self._get_scores_uncertainty_margin(),
            self._get_scores_uncertainty_entropy()
        ]

        scores = dict()
        for score, score_name in scores_with_names:
            scores[score_name] = score
        return scores

    """
    The following three uncertainty scores are taken from
    http://burrsettles.com/pub/settles.activelearning.pdf, Section 3.1, page 12f
    and also explained in https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b
    """
    def _get_scores_uncertainty_least_confidence(self):
        scores = np.array([1 - max(class_probabilities) for class_probabilities in self.model_output])
        return scores, "uncertainty_least_confidence"

    def _get_scores_uncertainty_margin(self):
        scores = 1 - _margin_largest_secondlargest(self.model_output)
        return scores, "uncertainty_margin"

    def _get_scores_uncertainty_entropy(self):
        scores = _entropy(self.model_output, axis=1)
        return scores, "uncertainty_entropy"
