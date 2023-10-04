import unittest

from PIL import Image

from lightly.transforms.equalize import HistogramEqualization


class TestRandomSolarization(unittest.TestCase):
    def test_on_pil_image(self) -> None:
        # without mask
        equalization = HistogramEqualization()
        for w in [32, 64, 128]:
            for h in [32, 64, 128]:
                sample = Image.new("RGB", (w, h))
                equalization(sample)

        # with masking
        for w in [32, 64, 128]:
            for h in [32, 64, 128]:
                equalization = HistogramEqualization(mask=Image.new("L", (w, h)))
                sample = Image.new("RGB", (w, h))
                equalization(sample)
