""" Keypoint """
from typing import Union, Tuple, List

import numpy as np

class KeypointDetection:
    """Class which represents all keypoints of one object.

    Attributes:
        keypoints:
            Predicted keypoints in the [x0, y0, c0, ... xk, yk, ck] format.
            Thus the shape of the array for k keypoints must be (k*3, ).
        object_id:
            The id of the predicted object, e.g. "3" denoting a person.
            This is not used at the moment.

    Examples:
        >>> # Create the representation of wo keypoints with confidences
        >>> # of 0.8 and 0.1 respectively.
        >>> keypoints = [334, 534, 0.8, 456, 432, 0.1]
        >>> keypoints = np.array(keypoints)
        >>> keypoint_detections = KeypointDetection(keypoints, 3)
    """

    def __init__(self, keypoints: np.ndarray, object_id: int = -1):
        self.keypoints = keypoints
        self.object_id = object_id
        self._format_check()

    def _format_check(self):
        if len(self.keypoints) % 3 != 0 or len(self.keypoints.shape) != 1:
            raise ValueError("keypoints must be in the format of "
                             "[x0, y0, c0, ... xk, yk, ck], but they are not.")
        if any(self.get_confidences() < 0):
            raise ValueError("Confidences contain values < 0.")
        if any(self.get_confidences() > 1):
            raise ValueError("Confidences contain values > 1.")

    def get_confidences(self) -> np.ndarray:
        confidences = self.keypoints[2::3]
        return confidences



class KeypointDetectionOutput:
    """Class which represents all keypoints detections in one images.

        Attributes:
            keypoint_detections:
                One KeypointDetection for each object having keypoints
                detected in the image

    """
    def __init__(self, keypoint_detections: List[KeypointDetection]):
        self.keypoint_detections = keypoint_detections
