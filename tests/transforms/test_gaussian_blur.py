import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter

from lightly.transforms.gaussian_blur import GaussianBlur


class TestGaussianBlur:
    @pytest.mark.parametrize("w", [1, 10, 100])
    @pytest.mark.parametrize("h", [1, 10, 100])
    def test_on_pil_image(self, w: int, h: int) -> None:
        gaussian_blur = GaussianBlur(prob=1.0)
        sample = Image.new("RGB", (w, h))
        out = gaussian_blur(sample)
        assert isinstance(out, Image.Image)
        assert out.size == (w, h)

    @pytest.mark.parametrize("w", [1, 10, 100])
    @pytest.mark.parametrize("h", [1, 10, 100])
    def test_on_tensor(self, w: int, h: int) -> None:
        gaussian_blur = GaussianBlur(prob=1.0)
        sample_tensor = torch.randn(3, h, w)
        out = gaussian_blur(sample_tensor)
        assert isinstance(out, torch.Tensor)
        assert out.shape == sample_tensor.shape
        assert out.dtype == sample_tensor.dtype

    @pytest.mark.parametrize("w", [1, 10, 100])
    @pytest.mark.parametrize("h", [1, 10, 100])
    def test_on_uint8_tensor(self, w: int, h: int) -> None:
        gaussian_blur = GaussianBlur(prob=1.0)
        sample_tensor = torch.randint(0, 256, (3, h, w), dtype=torch.uint8)
        out = gaussian_blur(sample_tensor)
        assert isinstance(out, torch.Tensor)
        assert out.shape == sample_tensor.shape
        assert out.dtype == torch.uint8

    def test_prob_zero_returns_original_without_conversion(self) -> None:
        gaussian_blur = GaussianBlur(prob=0.0)
        sample_tensor = torch.randn(3, 16, 16)
        out = gaussian_blur(sample_tensor)
        assert out is sample_tensor

    def test_sigma_zero_returns_original(self) -> None:
        gaussian_blur = GaussianBlur(prob=1.0, sigmas=(0.0, 0.0))
        sample_tensor = torch.randn(3, 16, 16)
        out = gaussian_blur(sample_tensor)
        assert out is sample_tensor

    def test_matches_pil_sigma_semantics(self) -> None:
        """Regression test for #1052: torchvision blur must match PIL GaussianBlur(radius=sigma)."""
        generator = torch.Generator().manual_seed(0)
        arr = torch.randint(
            0, 256, (64, 64, 3), dtype=torch.uint8, generator=generator
        ).numpy()
        pil_img = Image.fromarray(arr)

        for sigma in (0.5, 1.0, 2.0):
            pil_blurred = np.asarray(
                pil_img.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32
            )
            tv_blurred_img = GaussianBlur(prob=1.0, sigmas=(sigma, sigma))(pil_img)
            assert isinstance(tv_blurred_img, Image.Image)
            tv_blurred = np.asarray(tv_blurred_img, dtype=np.float32)
            mean_abs_diff = float(np.abs(pil_blurred - tv_blurred).mean())
            assert mean_abs_diff < 3.0

    def test_on_2d_grayscale_tensor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from torchvision.transforms import functional as F_v1

        import lightly.transforms.gaussian_blur as gaussian_blur_module

        sample = torch.rand(16, 16)
        out_default = GaussianBlur(prob=1.0, sigmas=(1.0, 1.0))(sample)
        assert isinstance(out_default, torch.Tensor)
        assert out_default.shape == (1, 16, 16)

        # Also verify against legacy torchvision v1 functional backend.
        monkeypatch.setattr(gaussian_blur_module, "F", F_v1)
        monkeypatch.setattr(gaussian_blur_module, "_TRANSFORMS_V2", False)
        out_v1 = GaussianBlur(prob=1.0, sigmas=(1.0, 1.0))(sample)
        assert isinstance(out_v1, torch.Tensor)
        assert out_v1.shape == (1, 16, 16)

    def test_preserves_cmyk_pil_mode_on_legacy_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from torchvision.transforms import functional as F_v1

        import lightly.transforms.gaussian_blur as gaussian_blur_module

        sample = Image.new("CMYK", (16, 16), (10, 20, 30, 40))
        out_default = GaussianBlur(prob=1.0, sigmas=(1.0, 1.0))(sample)
        assert isinstance(out_default, Image.Image)
        assert out_default.mode == "CMYK"

        monkeypatch.setattr(gaussian_blur_module, "F", F_v1)
        monkeypatch.setattr(gaussian_blur_module, "_TRANSFORMS_V2", False)
        out_v1 = GaussianBlur(prob=1.0, sigmas=(1.0, 1.0))(sample)
        assert isinstance(out_v1, Image.Image)
        assert out_v1.mode == "CMYK"

    @pytest.mark.parametrize("mode", ["RGBa", "La"])
    def test_preserves_premultiplied_alpha_pil_modes(
        self, mode: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from torchvision.transforms import functional as F_v1

        import lightly.transforms.gaussian_blur as gaussian_blur_module

        fill = (10, 20, 30, 255) if mode == "RGBa" else (128, 255)
        sample = Image.new(mode, (16, 16), fill)
        out_default = GaussianBlur(prob=1.0, sigmas=(1.0, 1.0))(sample)
        assert isinstance(out_default, Image.Image)
        assert out_default.mode == mode
        assert out_default.size == (16, 16)

        monkeypatch.setattr(gaussian_blur_module, "F", F_v1)
        monkeypatch.setattr(gaussian_blur_module, "_TRANSFORMS_V2", False)
        out_v1 = GaussianBlur(prob=1.0, sigmas=(1.0, 1.0))(sample)
        assert isinstance(out_v1, Image.Image)
        assert out_v1.mode == mode
        assert out_v1.size == (16, 16)

    def test_on_cpu_float16_tensor(self) -> None:
        sample = torch.rand(3, 16, 16, dtype=torch.float16)
        out = GaussianBlur(prob=1.0, sigmas=(1.0, 1.0))(sample)
        assert isinstance(out, torch.Tensor)
        assert out.shape == sample.shape
        assert out.dtype == torch.float16
        assert torch.isfinite(out).all()

    def test_init__keyword_only(self) -> None:
        with pytest.raises(TypeError):
            GaussianBlur(0.5)  # type: ignore[misc]
