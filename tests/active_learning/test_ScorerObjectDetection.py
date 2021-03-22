import unittest

from lightly.active_learning.utils.bounding_box import BoundingBox
from lightly.active_learning.utils.object_detection_output import ObjectDetectionOutput
from lightly.active_learning.scorers.detection import ScorerObjectDetection



class TestScorerObjectDetection(unittest.TestCase):

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


    def test_object_detection_scorer(self):

        # convert bounding boxes
        W, H = 128, 128
        for data in self.dummy_data:
            for i, box in enumerate(data['boxes']):
                x0 = box[0] / W
                y0 = box[1] / H
                x1 = box[2] / W
                y1 = box[3] / H
                data['boxes'][i] = BoundingBox(x0, y0, x1, y1)

        for i, data in enumerate(self.dummy_data):
            self.dummy_data[i] = ObjectDetectionOutput.from_class_probabilities(
                data['boxes'],
                data['object_probabilities'],
                data['class_probabilities'],
                data['labels'],
            )

        scorer = ScorerObjectDetection(self.dummy_data)
        scores = scorer._calculate_scores()

        res = scores['object-frequency']
        self.assertEqual(len(res), len(self.dummy_data))
        self.assertListEqual(res.tolist(), [1.0, 0.95, 0.9])

        res = scores['prediction-margin']
        self.assertEqual(len(res), len(self.dummy_data))
        self.assertListEqual(res.tolist(), [0.5514945, 0.9488, 0.])
