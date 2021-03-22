""" TODO """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import List

from lightly.active_learning.utils.bounding_box import BoundingBox


class ObjectDetectionOutput:
    """Class which unifies different object detection output formats.

    Attributes:
        boxes:
            List of BoundingBox objects with coordinates (x0, y0, x1, y1).
        scores:
            List of confidence scores (i.e. max(class prob) * objectness).
        labels:
            List of labels.

    Examples:
        >>> # typical model output
        >>> prediction = {
        >>>     'boxes': [[0.1, 0.2, 0.3, 0.4]],
        >>>     'scores': [0.1234],
        >>>     'labels': [1]
        >>> }
        >>>
        >>> # convert bbox to objects
        >>> boxes = [BoundingBox(0.1, 0.2, 0.3, 0.4)]
        >>> scores = prediction['scores']
        >>> labels = prediction['labels']
        >>>
        >>> # create detection output
        >>> detection_output = ObjectDetectionOutput(boxes, scores, labels)

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
        """Helper to convert from output format with class probabilities.

        Examples:
            >>> # typical model output
            >>> prediction = {
            >>>     'boxes': [[0.1, 0.2, 0.3, 0.4]],
            >>>     'object_probabilities': [0.6],
            >>>     'class_probabilities': [0.1, 0.5],
            >>>     'labels': [1]
            >>> }
            >>>
            >>> # convert bbox to objects
            >>> boxes = [BoundingBox(0.1, 0.2, 0.3, 0.4)]
            >>> object_probabilities = prediction['object_probabilities']
            >>> class_probabilities = prediction['class_probabilities']
            >>> labels = prediction['labels']
            >>>
            >>> # create detection output
            >>> detection_output = ObjectDetectionOutput.from_class_probabilities(
            >>>     boxes,
            >>>     object_probabilities,
            >>>     class_probabilities,
            >>>     labels
            >>> )

        """
        scores = []
        for o, c in zip(object_probabilities, class_probabilities):
            # calculate the score as the object probability times the maximum
            # of the class probabilities
            scores.append(o * max(c))

        return cls(boxes, scores, labels)