from typing import *

import numpy as np

from lightly.active_learning.scorers import ScorerClassification
from lightly.active_learning.scorers.scorer import Scorer
from lightly.active_learning.utils.object_detection_output import ObjectDetectionOutput


def _object_frequency(model_output: List[ObjectDetectionOutput],
                      frequency_penalty: float,
                      min_score: float) -> np.ndarray:
    """Score which prefers samples with many and diverse objects.

    Args:
        model_output:
            Predictions of the model of length N.
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
        for label in output.labels:
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


def _objectness_least_confidence(model_output: List[ObjectDetectionOutput]) -> np.ndarray:
    """Score which prefers samples with low max(class prob) * objectness.

    Args:
        model_output:
            Predictions of the model of length N.

    Returns:
        Numpy array of length N with the computed scores.

    """
    scores = []
    for output in model_output:
        if len(output.scores) > 0:
            # prediction margin is 1 - max(class probs), therefore the mean margin
            # is mean(1 - max(class probs)) which is 1 - mean(max(class probs))
            score = 1. - np.mean(output.scores)
        else:
            # set the score to 0 if there was no bounding box detected
            score = 0.
        scores.append(score)

    return np.asarray(scores)


def _reduce_classification_scores_over_boxes(
        model_output: List[ObjectDetectionOutput],
        reduce_fn_over_bounding_boxes: Callable[[np.ndarray], float] = np.max,
        default_value_no_bounding_box: float = 0
) -> Dict[str, List[float]]:
    """Calculates classification scores over the mean of all found objects

    Args:
        model_output:
            Predictions of the model of length N.
        reduce_fn_over_bounding_boxes:
            This function reduces the scores for each bounding box of an image
            to one score per image.
        default_value_no_bounding_box:
            This is the default score if the image does not have any bounding boxes found.

    Returns:
        Numpy array of length N with the computed scores.

    """
    # calculate a score dictionary for each sample
    scores_dict_list: List[dict[str, np.ndarray]] = []
    for index_sample, output in enumerate(model_output):
        probs = np.array(output.class_probabilities)
        scores_dict_this_sample = ScorerClassification(model_output=probs).calculate_scores()
        scores_dict_list.append(scores_dict_this_sample)

    score_names = ScorerClassification.score_names()

    # reduce it to one score per sample

    # Initialize the dictionary
    output_scores_dict = {score_name: [] for score_name in score_names}

    # Fill the dictionary
    for scores_dict in scores_dict_list:
        for score_name in score_names:
            score: np.ndarray = scores_dict[score_name]
            if len(score) > 0:
                scalar_score = float(reduce_fn_over_bounding_boxes(score))
            else:
                scalar_score = default_value_no_bounding_box
            output_scores_dict[score_name].append(scalar_score)

    return output_scores_dict


class ScorerObjectDetection(Scorer):
    """Class to compute active learning scores from the model_output of an object detection task.

    Currently supports the following scorers:

        `object_frequency`:
            This scorer uses model predictions to focus more on images which
            have many objects in them. Use this scorer if you want scenes
            with lots of objects in them like we usually want in
            computer vision tasks such as perception in autonomous driving.

        `objectness_least_confidence`:
            This score is 1 - the mean of the highest confidence prediction. Use this scorer
            to select images where the model is insecure about both whether it found an object
            at all and the class of the object.

        scores from ScorerClassification:
            These scores are computed for each object detection out of
            the class probability prediction for this detection.
            Then these scores are reduced to one score per image
            by taking the maximum. The scores are named as
            f"classification_{score_name}".

    Attributes:
        model_output:
            List of model outputs in an object detection setting.
        config:
            A dictionary containing additional parameters for the scorers.

            `frequency_penalty` (float):
                Used by the `object-frequency` scorer.
                If objects of the same class are within the same sample we
                multiply them with the penalty. 1.0 has no effect. 0.5 would
                count the first object fully and the second object of the same
                class only 50%. Lowering this value results in a more balanced
                setting of the classes. 0.0 is max penalty. (default: 0.25)
            `min_score` (float):
                Used by the `object-frequency` scorer.
                Specifies the minimum score per sample. All scores are
                scaled to [`min_score`, 1.0] range. Lowering the number makes
                the sampler focus more on samples with many objects.
                (default: 0.9)


    Examples:
        >>> # typical model output
        >>> predictions = [{
        >>>     'boxes': [[0.1, 0.2, 0.3, 0.4]],
        >>>     'object_probabilities': [0.1024],
        >>>     'class_probabilities': [[0.5, 0.41, 0.09]]
        >>> }]
        >>>
        >>> # generate detection outputs
        >>> model_output = []
        >>> for prediction in predictions:
        >>>     # convert each box to a BoundingBox object
        >>>     boxes = []
        >>>     for box in prediction['boxes']:
        >>>         x0, x1 = box[0], box[2]
        >>>         y0, y1 = box[1], box[3]
        >>>         boxes.append(BoundingBox(x0, y0, x1, y1))
        >>>     # create detection outputs
        >>>     output = ObjectDetectionOutput(
        >>>         boxes,
        >>>         prediction['object_probabilities'],
        >>>         prediction['class_probabilities']
        >>>     )
        >>>     model_output.append(output)
        >>>
        >>> # create scorer from output
        >>> scorer = ScorerObjectDetection(model_output)

    """

    def __init__(self,
                 model_output: List[ObjectDetectionOutput],
                 config: Dict = None):
        super(ScorerObjectDetection, self).__init__(model_output)
        self.config = config
        self._check_config()

    def _check_config(self):
        default_conf = {
            'frequency_penalty': 0.25,
            'min_score': 0.9
        }

        # Check if we have a config dictionary passed in constructor
        if self.config is not None and isinstance(self.config, dict):
            # check if constructor received keys which are wrong
            for k in self.config.keys():
                if k not in default_conf.keys():
                    raise KeyError(
                        f'Scorer config parameter {k} is not a valid key. '
                        f'Use one of: {default_conf.keys()}'
                    )

            # for now all values in config should be between 0.0 and 1.0 and numbers
            for k, v in self.config.items():
                if not (isinstance(v, float) or isinstance(v, int)):
                    raise ValueError(
                        f'Scorer config values must be numbers. However, '
                        f'{k} has a value of type {type(v)}.'
                    )

                if v < 0.0 or v > 1.0:
                    raise ValueError(
                        f'Scorer config parameter {k} value ({v}) out of range. '
                        f'Should be between 0.0 and 1.0.'
                    )

                # use default config if not specified in config
                for k, v in default_conf.items():
                    self.config[k] = self.config.get(k, v)
        else:
            self.config = default_conf

    @classmethod
    def score_names(cls) -> List[str]:
        """Returns the names of the calculated active learning scores
        """
        scorer = cls(model_output=[ObjectDetectionOutput([], [], [])])
        score_names = list(scorer.calculate_scores().keys())
        return score_names

    def calculate_scores(self) -> Dict[str, np.ndarray]:
        """Calculates and returns the active learning scores.

        Returns:
            A dictionary mapping from the score name (as string)
            to the scores (as a single-dimensional numpy array).
        """
        scores = dict()
        scores_with_names = [
            self._get_object_frequency(),
            self._get_objectness_least_confidence()
        ]
        for score, score_name in scores_with_names:
            score = np.nan_to_num(score)
            scores[score_name] = score

        # add classification scores
        scores_dict_classification = \
            _reduce_classification_scores_over_boxes(model_output=self.model_output)
        for score_name, score in scores_dict_classification.items():
            scores[score_name] = score

        return scores

    def _get_object_frequency(self):
        return _object_frequency(
            self.model_output,
            self.config['frequency_penalty'],
            self.config['min_score']), "object_frequency"

    def _get_objectness_least_confidence(self):
        return _objectness_least_confidence(self.model_output), "objectness_least_confidence"
