import unittest

from PIL import Image

from lightly.transforms.gaussian_blur import GaussianBlur


class TestGaussianBlur(unittest.TestCase):
    def test_on_pil_image(self) -> None:
        for w in range(1, 100):
            for h in range(1, 100):
                gaussian_blur = GaussianBlur()
                sample = Image.new("RGB", (w, h))
                gaussian_blur(sample)

    def test_raise_kernel_size_deprecation(self) -> None:
        gaussian_blur = GaussianBlur(kernel_size=2)
        self.assertWarns(DeprecationWarning)

    def test_raise_scale_deprecation(self) -> None:
        gaussian_blur = GaussianBlur(scale=0.1)
        self.assertWarns(DeprecationWarning)
