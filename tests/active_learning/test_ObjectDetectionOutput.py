import unittest

from lightly.active_learning.utils.bounding_box import BoundingBox
from lightly.active_learning.utils.object_detection_output import ObjectDetectionOutput


class TestObjectDetectionOutput(unittest.TestCase):

    def setUp(self):
        self.dummy_data = [
            {
                'boxes': [
                    [14, 16, 52, 85],
                    [58, 23, 124, 49]
                ],
                'object_probabilities': [
                    0.57573,
                    0.988
                ],
                'class_probabilities': [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.5, 0.1]
                ],
                'labels': [
                    0,
                    1,
                ]
            },
            {
                'boxes': [
                    [14, 16, 52, 85],
                ],
                'object_probabilities': [
                    0.1024,
                ],
                'class_probabilities': [
                    [0.5, 0.41, 0.09],
                ],
                'labels': [0]
            },
            {
                'boxes': [],
                'object_probabilities': [],
                'class_probabilities': [],
                'labels': []
            }
        ]

        # convert bounding boxes
        W, H = 128, 128
        for data in self.dummy_data:
            for i, box in enumerate(data['boxes']):
                x0 = box[0] / W
                y0 = box[1] / H
                x1 = box[2] / W
                y1 = box[3] / H
                data['boxes'][i] = BoundingBox(x0, y0, x1, y1)

    def test_object_detection_output(self):
        
        outputs_1 = []
        outputs_2 = []
        for i, data in enumerate(self.dummy_data):
            output = ObjectDetectionOutput(
                data['boxes'],
                data['object_probabilities'],
                data['class_probabilities'],
            )
            outputs_1.append(output)

            scores = [o * max(c) for o, c in zip(data['object_probabilities'], data['class_probabilities'])]
            output_2 = ObjectDetectionOutput.from_scores(
                data['boxes'],
                scores,
                data['labels'],
            )

        for output_1, output_2 in zip(outputs_1, outputs_2):
            for x, y in zip(output_1.labels, output_2.labels):
                self.assertEqual(x, y)
            for x, y in zip(output_1.scores, output_2.scores):
                self.assertEqual(x, y)


    def test_object_detection_output_illegal_args(self):

        with self.assertRaises(ValueError):
            # score > 1
            ObjectDetectionOutput.from_scores(
                [BoundingBox(0, 0, 1, 1)],
                [1.1],
                [0]
            )

        with self.assertRaises(ValueError):
            # score < 0
            ObjectDetectionOutput.from_scores(
                [BoundingBox(0, 0, 1, 1)],
                [-1.],
                [1]
            )

        with self.assertRaises(ValueError):
            # different length
            ObjectDetectionOutput(
                [BoundingBox(0, 0, 1, 1)],
                [0.5, 0.2],
                [1, 2]
            )

