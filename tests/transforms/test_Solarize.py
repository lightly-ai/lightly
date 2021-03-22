import unittest
from PIL import Image

from lightly.transforms.solarize import RandomSolarization


class TestRandomSolarization(unittest.TestCase):

    def test_on_pil_image(self):
        for w in [32,64,128]:
            for h in [32,64,128]:
                solarization = RandomSolarization(0.5)
                sample = Image.new('RGB', (w, h))
                solarization(sample)
