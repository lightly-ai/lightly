import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from lightly.transforms.solarize import RandomSolarization


class TestRandomSolarization:
    @pytest.mark.parametrize("w", [32, 64, 128])
    @pytest.mark.parametrize("h", [32, 64, 128])
    def test_on_pil_image(self, w: int, h: int) -> None:
        solarization = RandomSolarization(0.5)
        sample = Image.new("RGB", (w, h))
        out = solarization(sample)
        assert isinstance(out, Image.Image)
        assert out.size == (w, h)

    def test_pil_matches_imageops_solarize(self) -> None:
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        sample = Image.fromarray(arr)

        out = RandomSolarization(prob=1.0, threshold=128)(sample)
        expected = ImageOps.solarize(sample, threshold=128)
        assert isinstance(out, Image.Image)
        assert np.array_equal(np.asarray(out), np.asarray(expected))

    def test_on_uint8_tensor(self) -> None:
        sample = torch.tensor([[[0, 127, 128, 255]]], dtype=torch.uint8)
        out = RandomSolarization(prob=1.0, threshold=128)(sample)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.uint8
        assert torch.equal(out, torch.tensor([[[0, 127, 127, 0]]], dtype=torch.uint8))

    def test_on_float_tensor_with_int_threshold(self) -> None:
        sample = torch.tensor(
            [[[0.0, 127.0 / 255.0, 128.0 / 255.0, 1.0]]], dtype=torch.float32
        )
        out = RandomSolarization(prob=1.0, threshold=128)(sample)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
        expected = torch.tensor([[[0.0, 127.0 / 255.0, 1.0 - 128.0 / 255.0, 0.0]]])
        assert torch.allclose(out, expected)

    def test_on_float_tensor_with_float_threshold(self) -> None:
        sample = torch.tensor([[[0.0, 0.4, 0.6, 1.0]]], dtype=torch.float32)
        out = RandomSolarization(prob=1.0, threshold=0.5)(sample)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
        assert torch.allclose(out, torch.tensor([[[0.0, 0.4, 0.4, 0.0]]]))

    def test_prob_zero_returns_original(self) -> None:
        sample = torch.rand(3, 16, 16)
        out = RandomSolarization(prob=0.0)(sample)
        assert out is sample
