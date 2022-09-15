""" Active Learning Scorer for Semantic Segmentation """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved


from typing import Callable, Union, Generator, List, Dict

import numpy as np

from lightly.active_learning.scorers.scorer import Scorer
from lightly.active_learning.scorers import ScorerClassification


def _reduce_classification_scores_over_pixels(scores: np.ndarray,
                                              reduce_fn_over_pixels: Callable[[np.ndarray], float] = np.mean):
    """Reduces classification scores to a single floating point number.

    Args:
        scores:
            Numpy array of length N = W x H.
        reduce_fn_over_pixels:
            Function which reduces the scores in the array to a single float.

    Returns:
        A single floating point active learning score.

    """
    return float(reduce_fn_over_pixels(scores))


def _calculate_scores_for_single_prediction(prediction: np.ndarray):
    """Takes a single prediction array and calculates all scores for it.

    Args:
        prediction:
            The W x H x C array of predictions where C is the number of classes.

    Returns:
        A dictionary where each key is a score name and each item is the
        respective score (single float) for this prediction.

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

    Note: Since predictions for semantic segmentation may consume a lot of memory,
    it's also possible to use the scorer with a generator. See below for an example.

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
        >>> # use a generator of predictions to calculate active learning scores
        >>> def generator(filenames: List[string]):
        >>>     for filename in filenames:
        >>>         path = os.path.join(ROOT_PATH, filename)
        >>>         img_tensor = prepare_img(path).to('cuda') 
        >>>         with torch.no_grad():
        >>>             out = model(img_tensor)
        >>>             out = torch.softmax(out, axis=1)
        >>>             out = out.squeeze(0)
        >>>             out = out.transpose(0,2)
        >>>             yield out.cpu().numpy()
        >>>
        >>> # create a scorer and calculate the active learning scores
        >>> model_output_generator = generator(al_agent.query_set)
        >>> scorer = ScorerSemanticSegmentation(model_output_generator)
        >>> scores = scorer.calculate_scores()

    """

    def __init__(self,
                 model_output: Union[List[np.ndarray], Generator[np.ndarray, None, None]]):
        self.model_output = model_output

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
