""" Object Detection Outputs """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import List

from lightly.active_learning.utils.bounding_box import BoundingBox


class ObjectDetectionOutput:
    """Class which unifies different object detection output formats.

    Attributes:
        boxes:
            List of BoundingBox objects with coordinates (x0, y0, x1, y1).
        object_probabilities:
            List of probabilities that the boxes are indeed objects.
        class_probabilities:
            List of probabilities for the different classes for each box.
        scores:
            List of confidence scores (i.e. max(class prob) * objectness).
        labels:
            List of labels (i.e. argmax(class prob)).

    Examples:
        >>> # typical model output
        >>> prediction = {
        >>>     'boxes': [[0.1, 0.2, 0.3, 0.4]],
        >>>     'object_probabilities': [0.6],
        >>>     'class_probabilities': [0.1, 0.5],
        >>> }
        >>>
        >>> # convert bbox to objects
        >>> boxes = [BoundingBox(0.1, 0.2, 0.3, 0.4)]
        >>> object_probabilities = prediction['object_probabilities']
        >>> class_probabilities = prediction['class_probabilities']
        >>>
        >>> # create detection output
        >>> detection_output = ObjectDetectionOutput(
        >>>     boxes,
        >>>     object_probabilities,
        >>>     class_probabilities,
        >>> )

    """

    def __init__(self,
                 boxes: List[BoundingBox],
                 object_probabilities: List[float],
                 class_probabilities: List[List[float]]):

        if len(boxes) != len(object_probabilities) or \
            len(object_probabilities) != len(class_probabilities):
            raise ValueError('Boxes, object and class probabilities must be of '
                             f'same length but are {len(boxes)}, '
                             f'{len(object_probabilities)}, and '
                             f'{len(class_probabilities)}')

        scores = []
        labels = []
        for o, c in zip(object_probabilities, class_probabilities):
            # calculate the score as the object probability times the maximum
            # of the class probabilities
            scores.append(o * max(c))
            # the label is the argmax of the class probabilities
            labels.append(c.index(max(c)))

        self.boxes = boxes
        self.scores = scores
        self.labels = labels
        self.object_probabilities = object_probabilities
        self.class_probabilities = class_probabilities


    @classmethod
    def from_scores(cls,
                    boxes: List[BoundingBox],
                    scores: List[float],
                    labels: List[int]):
        """Helper to convert from output format with scores.

        We advise not using this method if you want to use the uncertainty
        active learning scores correctly.

        Since this output format does not provide class probabilities, they
        will be replaced by a estimated class probability computed by the
        objectness. The highest class probability matches the label.
        The objectness will be set to the score for each bounding box.

        Args:
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
            >>> detection_output = ObjectDetectionOutput.from_scores(
            >>>     boxes, scores, labels)

        """

        if any([score > 1 for score in scores]):
            raise ValueError('Scores must be smaller than or equal to one!')

        if any([score < 0 for score in scores]):
            raise ValueError('Scores must be larger than or equal to zero!')

        if not all([isinstance(label, int) for label in labels]):
            raise ValueError('Labels must be list of integers.')

        # create fake object probabilities
        object_probabilities = [s for s in scores]

        # create one-hot class probabilities
        max_label = max(labels) if len(labels) > 0 else 0
        class_probabilities = []
        for object_prob, label in zip(object_probabilities, labels):
            num_classes = max(max_label, 1)
            c = [(1 - object_prob) / num_classes] * (num_classes + 1)
            c[label] = object_prob
            class_probabilities.append(c)

        # create object detection output
        output = cls(boxes, object_probabilities, class_probabilities)
        output.scores = scores
        output.labels = labels
        return output