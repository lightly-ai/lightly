""" Keypoint """
from typing import Union


class Keypoint:
    """Class which represents a keypoint in an image.


    The x, y and occluded attributes are currently not used by the
    ScorerKeypointDetection, but might be in the future.
    
    Attributes:
        x:
            x coordinate of the keypoint
        y:
            y coordinate of the keypoint
        confidence:
            confidence of the keypoint, must be in [0,1]
        occluded: one of {None, True, False}
            None: unnknown
            True: keypoint is occluded
            False: keypoints is visible
            
            Some frameworks don't include an occlusion flag,
            thus this is optional. 

    Examples:
        >>> # Create a keypoint without knowing whether it is occluded.
        >>> keypoint = Keypoint(312, 413, 0.5)
        >>>
        >>> # Create a keypoint without knowing which is occluded.
        >>> keypoint = Keypoint(312, 413, 0.5, occluded=True)
        >>>

    """

    def __init__(self, x: Union[float, int], y: Union[float, int],
                 confidence: float, occluded: Union[bool, None] = None):
        self.x = x
        self.y = y
        if confidence < 0 or confidence > 1:
            raise ValueError(
                f"Confidence must be in [0, 1], but is {confidence}.")
        self.confidence = confidence
        self.occluded = occluded
