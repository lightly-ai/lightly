import unittest

from lightly.active_learning.utils.keypoint import Keypoint


class TestKeypoint(unittest.TestCase):

    def test_keypoint_valid(self):
        for x in [123, 123.]:
            for y in [456, 456.]:
                for confidence in [0, 0.5, 1]:
                    for occluded in [None, True, False]:
                        with self.subTest(
                                x=x, y=y,
                                confidence=confidence, occluded=occluded
                        ):
                            if occluded is not None:
                                Keypoint(x, y, confidence, occluded)
                            else:
                                Keypoint(x, y, confidence)

    def test_keypoint_invalid_confidence(self):
        for x in [123, 123.]:
            for y in [456, 456.]:
                for confidence in [-1, 2]:
                    for occluded in [None, True, False]:
                        with self.subTest(
                                x=x, y=y,
                                confidence=confidence, occluded=occluded
                        ), self.assertRaises(ValueError):
                            if occluded is not None:
                                Keypoint(x, y, confidence, occluded)
                            else:
                                Keypoint(x, y, confidence)


