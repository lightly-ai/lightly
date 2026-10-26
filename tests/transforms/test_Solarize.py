import pytest
from PIL import Image

from lightly.transforms.solarize import RandomSolarization


class TestRandomSolarization:
    @pytest.mark.parametrize("w", [32, 64, 128])
    @pytest.mark.parametrize("h", [32, 64, 128])
    def test_on_pil_image(self, w: int, h: int) -> None:
        solarization = RandomSolarization(0.5)
        sample = Image.new("RGB", (w, h))
        solarization(sample)
