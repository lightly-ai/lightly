import warnings
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

        The following three uncertainty scores are taken from
        http://burrsettles.com/pub/settles.activelearning.pdf, Section 3.1, page 12f
        and also explained in https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b
        They all have in common, that the score is highest if all classes have the
        same confidence and are 0 if the model assigns 100% probability to a single class.
        The differ in the number of class confidences they take into account.

        `uncertainty_least_confidence`:
            This score is 1 - the highest confidence prediction. It is high
            when the confidence about the most probable class is low.

        `uncertainty_margin`
            This score is 1- the margin between the highest conficence
            and second highest confidence prediction. It is high when the model
            cannot decide between the two most probable classes.

        `uncertainty_entropy`:
            This scorer computes the entropy of the prediction. The confidences
             for all classes are considered to compute the entropy of a sample.

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
    def __init__(self, model_output: Union[np.ndarray, List[List[float]]]):
        if not isinstance(model_output, np.ndarray):
            model_output = np.array(model_output)

        validated_model_output = self.ensure_valid_model_output(model_output)

        super(ScorerClassification, self).__init__(validated_model_output)

    def ensure_valid_model_output(self, model_output: np.ndarray) -> np.ndarray:
        if len(model_output) == 0:
            return model_output
        if len(model_output.shape) != 2:
            raise ValueError("ScorerClassification model_output must be a 2-dimensional array")
        if model_output.shape[1] == 0:
            raise ValueError("ScorerClassification model_output must not have an empty dimension 1")
        if model_output.shape[1] == 1:
            # assuming a binary classification problem with
            # the model_output denoting the probability of the first class
            model_output = np.concatenate([model_output, 1-model_output], axis=1)
        return model_output

    @classmethod
    def score_names(cls) -> List[str]:
        """Returns the names of the calculated active learning scores
        """
        score_names = list(cls(model_output=[[0.5, 0.5]]).calculate_scores().keys())
        return score_names

    def calculate_scores(self, normalize_to_0_1: bool = True) -> Dict[str, np.ndarray]:
        """Calculates and returns the active learning scores.

        Args:
            normalize_to_0_1:
                If this is true, each score is normalized to have a
                theoretical minimum of 0 and a theoretical maximum of 1.

        Returns:
            A dictionary mapping from the score name (as string)
            to the scores (as a single-dimensional numpy array).
        """
        if len(self.model_output) == 0:
            return {score_name: np.array([]) for score_name in self.score_names()}

        scores_with_names = [
            self._get_scores_uncertainty_least_confidence(),
            self._get_scores_uncertainty_margin(),
            self._get_scores_uncertainty_entropy()
        ]

        scores = dict()
        for score, score_name in scores_with_names:
            score = np.nan_to_num(score)
            scores[score_name] = score

        if normalize_to_0_1:
            scores = self.normalize_scores_0_1(scores)

        return scores

    def normalize_scores_0_1(self, scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        num_classes = self.model_output.shape[1]
        model_output_very_sure = np.zeros(shape=(1,num_classes))
        model_output_very_sure[0, 0] = 1
        model_output_very_unsure = np.ones_like(model_output_very_sure)/num_classes

        scores_minimum = ScorerClassification(model_output_very_sure).calculate_scores(normalize_to_0_1=False)
        scores_maximum = ScorerClassification(model_output_very_unsure).calculate_scores(normalize_to_0_1=False)

        for score_name in scores.keys():
            interp_xp = [float(scores_minimum[score_name]), float(scores_maximum[score_name])]
            interp_fp = [0, 1]
            scores[score_name] = np.interp(scores[score_name], interp_xp, interp_fp)

        return scores

    def _get_scores_uncertainty_least_confidence(self):
        scores = 1 - np.max(self.model_output, axis=1)
        return scores, "uncertainty_least_confidence"

    def _get_scores_uncertainty_margin(self):
        scores = 1 - _margin_largest_secondlargest(self.model_output)
        return scores, "uncertainty_margin"

    def _get_scores_uncertainty_entropy(self):
        scores = _entropy(self.model_output, axis=1)
        return scores, "uncertainty_entropy"
