from PIL import Image
import unittest

from lightly.transforms.simclr_transform import SimCLRTransform, SimCLRViewTransform


class TestSimCLRTransform(unittest.TestCase):

    def test_view_on_pil_image(self):
        for w in range(1, 10):
            for h in range(1, 10):
                single_view_transform = SimCLRViewTransform()
                sample = Image.new('RGB', (w, h))
                single_view_transform(sample)

    def test_multi_view_on_pil_image(self):
        for w in range(1, 10):
            for h in range(1, 10):
                multi_view_transform = SimCLRTransform()
                sample = Image.new('RGB', (w, h))
                multi_view_transform(sample)
                