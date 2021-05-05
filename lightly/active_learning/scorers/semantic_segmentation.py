""" Active Learning Scorer for Semantic Segmentation """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved


from typing import Union, Generator, List

import numpy as np

from lightly.active_learning.scorers.scorer import Scorer
from lightly.active_learning.scorers import ScorerClassification



def _reduce_classification_scores_over_pixels(scores: np.ndarray,
                                              reduce_fn_over_pixels: Callable[[np.ndarray], float] = np.mean):
    """TODO

    """
    return float(reduce_fn_over_pixels(scores))


def _calculate_scores_for_single_prediction(prediction: np.ndarray):
    """TODO

    """
    if len(prediction.shape) != 3:
        raise ValueError(
            'Invalid shape for semantic segmentation prediction! Expected '
            f'input of shape W x H x C but got {prediction.shape}.'
        )

    # reshape the W x H x C prediction into (W x H) x C
    w, h, c = prediction.shape
    prediction_flat = prediction.reshape(w * h, c)

    # calculate the scores
    classification_scorer = ScorerClassification(prediction_flat)

    # initialize dictionary to store results
    scores_dict = {}

    # reduce over pixels
    for score_name, scores in classification_scorer.calculate_scores().items():
        scores_dict[score_name] = \
            _reduce_classification_scores_over_pixels(scores)

    return scores_dict


class ScorerSemanticSegmentation(Scorer):
    """Class to compute active learning scores for a semantic segmentation task.

    Currently supports the following scores:
        `uncertainty scores`:
            These scores are calculated by treating each pixel as its own 
            classification task and taking the average of the classification
            uncertainty scores.

    Attributes:
        model_output:
            List or generator of semantic segmentation predictions. Each
            prediction should be of shape W x H x C, where C is the number 
            of classes (e.g. C=2 for two classes foreground and background).

    Examples:
        >>> # typical output of a semantic segmentation model is a list 
        >>> # of W x H x C prediction arrays (one for each input)
        >>> model_output[0].shape
        >>> > 512 x 512 x 2
        >>>
        >>> # create a scorer and calculate the active learning scores
        >>> scorer = ScorerSemanticSegmentation(model_output)
        >>> scorer.calculate_scores()

    """

    def __init__(self,
                 model_output: Union[List[np.ndarray], Generator[np.ndarray]]):
        super(ScorerSemanticSegmentation, self).__init__(model_output)

    def calculate_scores(self) -> Dict[str, np.ndarray]:
        """Calculates and returns the active learning scores.

        Returns:
            A dictionary mapping from the score name (as string) to the scores
            (as a single-dimensional numpy array).

        """
        scores = {}
        # iterate over list or generator of model outputs
        # careful! we can only iterate once if it's a generator
        for prediction in self.model_output:

            # get all active learning scores for this prediction
            # scores_ is a dictionary where each key is a score name and each 
            # item is a floating point number indicating the score
            scores_ = _calculate_scores_for_single_prediction(prediction)

            # append the scores for this prediction to the lists of scores
            for score_name, score in scores_.items():
                if score_name in scores:
                    scores[score_name].append(score)
                else:
                    scores[score_name] = [score]

        # make sure all returned lists are numpy arrays
        for score_name, score_list in scores.items():
            scores[score_name] = np.array(score_list)

        return scores
