""" Keypoint """
import json
from typing import Union, Tuple, List, Dict

import numpy as np


class KeypointInstancePrediction:
    """Class which represents all keypoints of one instance.

    Attributes:
        keypoints:
            Predicted keypoints in the [x0, y0, c0, ... xk, yk, ck] format.
        category_id:
            The id of the category of the object, e.g. "3" denoting a person.
            This is not used at the moment.
        score:
            An overall score for the prediction.
            This is not used at the moment.

    Examples:
        >>> # Create the representation of two keypoints with confidences
        >>> # of 0.8 and 0.1 respectively.
        >>> keypoints = [334, 534, 0.8, 456, 432, 0.1]
        >>> keypoint_detections = KeypointInstancePrediction(keypoints, 3)

    """

    def __init__(self, keypoints: List[float], category_id: int = -1,
                 score: float = -1.):
        self.keypoints = keypoints
        self.category_id = category_id
        self.score = score
        self._format_check()

    @classmethod
    def from_dict(cls, dict_: Dict[str, Union[int, List[float], float]]):
        category_id = dict_['category_id']
        keypoints = dict_['keypoints']
        score = dict_['score']
        return cls(keypoints=keypoints, category_id=category_id, score=score)

    def _format_check(self):
        """Raises a ValueError if the format is not as required.
        """
        if not isinstance(self.category_id, int):
            raise ValueError(
                f"Category_id must be an int, but is a {type(self.category_id)}")
        if not isinstance(self.score, float):
            raise ValueError(
                f"Score must be a float, but is a {type(self.score)}")

        if len(self.keypoints) % 3 != 0:
            raise ValueError("Keypoints must be in the format of "
                             "[x0, y0, c0, ... xk, yk, ck].")
        confidences = self.get_confidences()
        if any(c < 0 for c in confidences):
            raise ValueError("Confidences contain values < 0.")
        if any(c > 1 for c in confidences):
            raise ValueError("Confidences contain values > 1.")

    def get_confidences(self) -> List[float]:
        """Returns the confidence of each keypoint

        """
        confidences = self.keypoints[2::3]
        return confidences


class KeypointPrediction:
    """Class which represents all keypoints detections in one images.

        Attributes:
            keypoint_instance_predictions:
                One KeypointInstancePrediction for each instance having keypoints
                detected in the image

    """

    def __init__(self, keypoint_instance_predictions: List[
        KeypointInstancePrediction]):
        self.keypoint_instance_predictions = keypoint_instance_predictions

    @classmethod
    def from_dicts(cls,
                   dicts: List[Dict[str, Union[int, List[float], float]]]
                   ):
        keypoint_instance_predictions = [
            KeypointInstancePrediction.from_dict(dict_) for dict_ in dicts
        ]
        return cls(keypoint_instance_predictions)

    @classmethod
    def from_json_string(cls, json_string: str):
        dicsts = json.loads(json_string)
        return cls.from_dicts(dicsts)
