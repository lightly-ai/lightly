import unittest

from PIL import Image

from lightly.transforms.jigsaw import Jigsaw


class TestJigsaw(unittest.TestCase):
    def test_on_pil_image(self) -> None:
        crop = Jigsaw()
        sample = Image.new("RGB", (255, 255))
        crop(sample)
