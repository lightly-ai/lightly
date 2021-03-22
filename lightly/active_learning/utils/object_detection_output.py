""" TODO """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import List

from lightly.active_learning.utils.bounding_box import BoundingBox


class ObjectDetectionOutput:
    """TODO

    """

    def __init__(self,
                 boxes: List[BoundingBox],
                 scores: List[float],
                 labels: List[int]):

        if len(boxes) != len(scores) or len(scores) != len(labels):
            raise ValueError('Boxes, scores, and labels must be of same length '
                             f'but are {len(boxes)}, {len(scores)}, and '
                             f'{len(labels)}')

        if any([score > 1 for score in scores]):
            raise ValueError('Scores must be smaller than or equal to one!')

        if any([score < 0 for score in scores]):
            raise ValueError('Scores must be larger than or equal to zero!')

        self.boxes = boxes
        self.scores = scores
        self.labels = labels

    @classmethod
    def from_class_probabilities(cls,
                                 boxes: List[BoundingBox],
                                 object_probabilities: List[float],
                                 class_probabilities: List[List[float]],
                                 labels: List[int]):
        """TODO

        """

        scores = []
        for o, c in zip(object_probabilities, class_probabilities):
            # calculate the score as the object probability times the maximum
            # of the class probabilities
            scores.append(o * max(c))

        return cls(boxes, scores, labels)
