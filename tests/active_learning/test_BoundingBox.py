import unittest

from lightly.active_learning.utils.bounding_box import BoundingBox


class TestBoundingBox(unittest.TestCase):

    def test_bounding_box(self):
        bbox = BoundingBox(0.2, 0.3, 0.5, 0.6)

    def test_bounding_box_illogical_argument(self):
        with self.assertRaises(ValueError):
            # let x1 < x0
            bbox = BoundingBox(0.5, 0.3, 0.1, 0.6)

    def test_bounding_box_oob_arguments(self):
        with self.assertRaises(ValueError):
            bbox = BoundingBox(20, 30, 100, 200)

    def test_bounding_box_width(self):
        bbox = BoundingBox(0.2, 0.3, 0.5, 0.6)
        self.assertEqual(bbox.width, 0.3)

    def test_bounding_box_height(self):
        bbox = BoundingBox(0.2, 0.3, 0.5, 0.6)
        self.assertEqual(bbox.height, 0.3)

    def test_bounding_box_area(self):
        bbox = BoundingBox(0.2, 0.3, 0.5, 0.6)
        self.assertEqual(bbox.area, 0.09)
