import pytest
import torch
from PIL import Image as PILImageModule
from PIL.Image import Image as PILImage
from torch.testing import assert_close

from lightly.utils.dependency import torchvision_transforms_v2_available

if not torchvision_transforms_v2_available():
    pytest.skip("torchvision.transforms.v2 not available", allow_module_level=True)
from torchvision.tv_tensors import BoundingBoxes, Mask

from lightly.transforms import AddGridTransform


@pytest.fixture
def img_orig() -> PILImage:
    img = PILImageModule.new("RGB", (7, 5))
    return img


# ignore typing due to Any type used in torchvison.transforms.v2.Transform
@pytest.fixture
def bbox_expected() -> BoundingBoxes:  # type: ignore[misc]
    bbox = torch.tensor(
        [
            [0.00, 0.00, 1.66, 3.50],
            [0.00, 3.50, 1.66, 7.00],
            [1.66, 0.00, 3.33, 3.50],
            [1.66, 3.50, 3.33, 7.00],
            [3.33, 0.00, 5.00, 3.50],
            [3.33, 3.50, 5.00, 7.00],
        ]
    )
    return BoundingBoxes(bbox, canvas_size=(7, 5), format="XYXY")


# ignore typing due to Any type used in torchvison.transforms.v2.Transform
@pytest.fixture
def mask_expected() -> Mask:  # type: ignore[misc]
    mask = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0, 2.0],
            [0.0, 0.0, 1.0, 1.0, 2.0],
            [0.0, 0.0, 1.0, 1.0, 2.0],
            [0.0, 0.0, 1.0, 1.0, 2.0],
            [3.0, 3.0, 4.0, 4.0, 5.0],
            [3.0, 3.0, 4.0, 4.0, 5.0],
            [3.0, 3.0, 4.0, 4.0, 5.0],
        ]
    ).to(torch.int64)
    return Mask(mask)


@pytest.fixture
def cols() -> int:
    return 3


@pytest.fixture
def rows() -> int:
    return 2


@pytest.fixture
def img_h() -> int:
    return 7


@pytest.fixture
def img_w() -> int:
    return 5


def test_AddGridTransform_bbox(
    bbox_expected: BoundingBoxes, cols: int, rows: int, img_h: int, img_w: int
) -> None:
    tr = AddGridTransform(rows, cols)
    bbox_empty = BoundingBoxes(
        torch.zeros(1, 4), format="XYXY", canvas_size=(img_h, img_w)
    )
    bbox_tr = tr(bbox_empty)
    assert_close(bbox_expected, bbox_tr, atol=0.01, rtol=0.01)


def test_AddGridTransform_mask(mask_expected: Mask, cols: int, rows: int) -> None:
    tr = AddGridTransform(rows, cols)
    mask_empty = Mask(torch.randint(0, 1, mask_expected.shape[-2:]).to(torch.int64))
    mask_tr = tr(mask_empty)
    assert (mask_tr == mask_expected).all()


def test_AddGridTransform_as_dict(img_orig: PILImage, cols: int, rows: int) -> None:
    sample = {
        "img": img_orig,
        "bbox": BoundingBoxes(
            torch.randn(1, 4), canvas_size=img_orig.size, format="XYXY"
        ),
        "mask": Mask(torch.randn(img_orig.size).to(torch.int64)),
    }
    sample_tr = AddGridTransform(rows, cols)(sample)
    assert sample_tr["img"] == img_orig
    assert sample_tr["bbox"].shape == (rows * cols, 4)
    assert (
        sample_tr["mask"].unique() == torch.linspace(0, rows * cols - 1, rows * cols)
    ).all()


def test_AddGridTransform_as_args(img_orig: PILImage, cols: int, rows: int) -> None:
    bbox_empty = BoundingBoxes(
        torch.zeros(1, 4), format="XYXY", canvas_size=img_orig.size
    )
    out = AddGridTransform(rows, cols)(img_orig, bbox_empty)
    assert len(out) == 2
