import pytest
import torch

from lightly.utils.dependency import torchvision_transforms_v2_available

if not torchvision_transforms_v2_available():
    pytest.skip("torchvision.transforms.v2 not available", allow_module_level=True)
from torchvision.tv_tensors import Image, Mask

from lightly.transforms.detcon_transform import (
    DetConSimCLRViewTransform,
    DetConSTransform,
)


# ignore typing due to Any type used in torchvison.transforms.v2.Transform
@pytest.fixture()
def img() -> Image:  # type: ignore[misc]
    return Image(torch.randn(3, 256, 256))


# ignore typing due to Any type used in torchvison.transforms.v2.Transform
@pytest.fixture()
def mask() -> Mask:  # type: ignore[misc]
    return Mask(torch.randint(0, 4, (1, 256, 256)), dtype=torch.int64)


class TestDetConSimCLRViewTransform:
    def test_given_masks(self, img: Image, mask: Mask) -> None:
        tr = DetConSimCLRViewTransform(input_size=(224, 224))

        img_tr, mask_tr = tr(img, mask)
        assert img_tr.shape == (3, 224, 224)
        assert mask_tr.shape == (1, 224, 224)


class TestDetConTransform:
    def test_given_masks(self, img: Image, mask: Mask) -> None:
        # deactivate anything that could change the mask
        tr = DetConSTransform(input_size=(256, 256), min_scale=1.0, rr_prob=0.0)

        (img_tr1, mask_tr1), (img_tr2, mask_tr2) = tr(img, mask)

        assert img_tr1.shape == (3, 256, 256)
        assert mask_tr1.shape == (1, 256, 256)
        assert img_tr2.shape == (3, 256, 256)
        assert mask_tr2.shape == (1, 256, 256)

        assert (mask_tr1.unique() == torch.tensor([0, 1, 2, 3])).all()

    def test_generate_grid_mask(self, img: Image, mask: Mask) -> None:
        # deactivate anything that could change the mask
        tr = DetConSTransform(
            grid_size=(4, 4), input_size=(256, 256), min_scale=1.0, rr_prob=0.0
        )

        (img_tr1, mask_tr1), (img_tr2, mask_tr2) = tr(img, mask)

        assert img_tr1.shape == (3, 256, 256)
        assert mask_tr1.shape == (1, 256, 256)
        assert img_tr2.shape == (3, 256, 256)
        assert mask_tr2.shape == (1, 256, 256)

        assert (mask_tr1.unique() == torch.arange(4 * 4)).all()
