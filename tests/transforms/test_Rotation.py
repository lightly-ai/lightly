from PIL import Image
import unittest

from lightly.transforms.rotation import RandomRotate, RandomRotateDegrees, random_rotation_transform


class TestRotation(unittest.TestCase):

    def test_RandomRotate_on_pil_image(self):
        for w in range(1, 100):
            for h in range(1, 100):
                random_rotate = RandomRotate()
                sample = Image.new('RGB', (w, h))
                random_rotate(sample)
    
    def test_RandomRotateDegrees_on_pil_image(self):
        for w in range(1, 10):
            for h in range(1, 10):
                for r in range(1, 90):
                    random_rotate = RandomRotateDegrees(prob=0.5, degrees=r)
                    sample = Image.new('RGB', (w, h))
                    random_rotate(sample)

    def test_random_rotation_transform(self):
        transform = random_rotation_transform(rr_prob=1.0, rr_degrees=None)
        assert isinstance(transform, RandomRotate)
        transform = random_rotation_transform(rr_prob=1.0, rr_degrees=45)
        assert isinstance(transform, RandomRotateDegrees)
        transform = random_rotation_transform(rr_prob=1.0, rr_degrees=(30, 45))
        assert isinstance(transform, RandomRotateDegrees)
