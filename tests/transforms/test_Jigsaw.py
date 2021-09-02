from PIL import Image
import unittest

from lightly.transforms import Jigsaw


class TestJigsaw(unittest.TestCase):

    def test_on_pil_image(self):
        crop = Jigsaw()
        sample = Image.new('RGB', (255,255))
        crop(sample)
