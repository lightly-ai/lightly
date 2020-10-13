from PIL import Image
import unittest

from lightly.transforms import GaussianBlur


class TestGaussianBlur(unittest.TestCase):

    def test_on_pil_image(self):
        for w in range(1, 100):
            for h in range(1, 100):
                gaussian_blur = GaussianBlur(w * 0.1)
                sample = Image.new('RGB', (w, h))
                gaussian_blur(sample)
