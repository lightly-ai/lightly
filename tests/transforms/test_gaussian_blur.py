import pytest
import torch
from PIL import Image

from lightly.transforms.gaussian_blur import GaussianBlur


class TestGaussianBlur:
    @pytest.mark.parametrize("w", [1, 10, 100])
    @pytest.mark.parametrize("h", [1, 10, 100])
    def test_on_pil_image(self, w: int, h: int) -> None:
        gaussian_blur = GaussianBlur()
        sample = Image.new("RGB", (w, h))
        gaussian_blur(sample)

    @pytest.mark.parametrize("w", [1, 10, 100])
    @pytest.mark.parametrize("h", [1, 10, 100])
    def test_on_tensor(self, w: int, h: int) -> None:
        gaussian_blur = GaussianBlur()
        sample_tensor = torch.randn(3, h, w)
        gaussian_blur(sample_tensor)

    def test_init__keyword_only(self) -> None:
        with pytest.raises(TypeError):
            GaussianBlur(0.5)  # type: ignore[misc]
