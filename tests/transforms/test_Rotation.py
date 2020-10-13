from PIL import Image
import unittest

from lightly.transforms import RandomRotate


class TestGaussianBlur(unittest.TestCase):

    def test_on_pil_image(self):
        for w in range(1, 100):
            for h in range(1, 100):
                random_rotate = RandomRotate()
                sample = Image.new('RGB', (w, h))
                random_rotate(sample)
