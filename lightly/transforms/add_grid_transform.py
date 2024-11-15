# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
from math import ceil
from typing import Any, Dict, Tuple

import torch
from torchvision.transforms.v2 import CenterCrop, ConvertBoundingBoxFormat, Transform
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Mask

from lightly.utils import dependency as _dependency


# ignore typing due to Any type used in torchvison.transforms.v2.Transform
class AddGridTransform(Transform):  # type: ignore[misc]
    """Implements the naive segmentation into a regular grid from DetCon. [0]_

    Input to this transform:
        Any datastructure containing one or several `torchvision.tv_tensor.BoundingBoxes`
        and/or `torchvision.tv_tensor.Mask`, such as tuples or arbitrarily nested dictionaries.
        For all supported data structures check [1]_. Masks should be of size (*, H, W) and
        BoundingBoxes can be of arbitrary shape.

    Output of this transform:
        Leaves any images in the data structure untouched, but overwrites any bounding
        boxes and masks by a regular grid. Bounding boxes will take shape (num_rows*num_cols, 4)
        and masks will be of shape (1, H, W) with integer values in the range [0, num_rows*num_cols-1].

    Example::

        img = torch.randn((3, 16, 16))
        bboxes = BoundingBoxes(torch.randn((1, 4)), format="XYXY", canvas_size=img.shape[-2:])
        mask = Mask(torch.randn((16, 16)).to(torch.int64))
        tr = AddGridTransform(num_rows=4, num_cols=4)
        # the image will be untouched the bounding boxes will be a regular grid
        img, bboxes, mask = tr(img, bboxes, mask)
        # bboxes of shape (num_rows*num_cols, 4)
        # mask with value range [0, num_rows*num_cols-1]

    References:
        .. [0] DetCon, 2021, https://arxiv.org/abs/2103.10957
        .. [1] torchvision Getting started with transforms v2, https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html

    Attributes:
        num_rows: number of grid rows in the segmentation
        num_cols: number of grid columns in the segmentation
    """

    def __init__(self, num_rows: int, num_cols: int) -> None:
        if not _dependency.torchvision_transforms_v2_available():
            raise ImportError(
                "AddGridTransform requires torchvision.transforms.v2, included in torchvision>=0.17"
            )
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, BoundingBoxes):
            return _create_bounding_boxes_grid(
                num_rows=self.num_rows,
                num_cols=self.num_cols,
                canvas_size=inpt.canvas_size,
                dtype=inpt.dtype,
                device=inpt.device,
                requires_grad=inpt.requires_grad,
                format=inpt.format,
            )
        elif isinstance(inpt, Mask):
            if inpt.dim() < 2:
                raise ValueError(
                    f"Expected mask to have at least 2 dimensions, got {inpt.dim()} instead."
                )
            return _create_mask_grid(
                num_rows=self.num_rows,
                num_cols=self.num_cols,
                canvas_size=inpt.shape[-2:],
                dtype=inpt.dtype,
                device=inpt.device,
                requires_grad=inpt.requires_grad,
            )
        else:
            return inpt


def _create_bounding_boxes_grid(
    num_rows: int,
    num_cols: int,
    canvas_size: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    format: BoundingBoxFormat,
) -> BoundingBoxes:
    h, w = canvas_size

    xs = torch.linspace(0, w, num_cols + 1)
    ys = torch.linspace(0, h, num_rows + 1)

    x_min, x_max = xs[:-1], xs[1:]
    y_min, y_max = ys[:-1], ys[1:]

    x_min, y_min = torch.meshgrid(x_min, y_min, indexing="ij")
    x_max, y_max = torch.meshgrid(x_max, y_max, indexing="ij")

    bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    bboxes = BoundingBoxes(
        bboxes.view(-1, 4),
        format="XYXY",
        canvas_size=canvas_size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    return ConvertBoundingBoxFormat(format)(bboxes)


def _create_mask_grid(
    num_rows: int,
    num_cols: int,
    canvas_size: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> Mask:
    h, w = canvas_size

    patch_h = ceil(h / num_rows)
    patch_w = ceil(w / num_cols)

    classes = torch.linspace(0, num_rows * num_cols - 1, num_rows * num_cols).to(
        torch.int64
    )
    patches = classes[:, None, None].expand(-1, patch_h, patch_w)
    patches = patches.view(num_rows, num_cols, patch_h, patch_w)
    mask = (
        patches.permute(0, 2, 1, 3)
        .contiguous()
        .view(num_rows * patch_h, num_cols * patch_w)
    )
    mask = mask.unsqueeze(0)
    mask = CenterCrop((canvas_size))(mask)

    return Mask(mask, dtype=dtype, device=device, requires_grad=requires_grad)
