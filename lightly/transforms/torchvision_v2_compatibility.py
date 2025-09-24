#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Union

import torch
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import ToTensor as ToTensorV1

from lightly.utils import dependency as _dependency

if _dependency.torchvision_transforms_v2_available():
    from torchvision.transforms import v2 as torchvision_transforms

    _TRANSFORMS_V2 = True
else:
    from torchvision import transforms as torchvision_transforms

    _TRANSFORMS_V2 = False


def ToTensor() -> Union[torchvision_transforms.Compose, ToTensorV1]:
    """Convert a PIL Image to a tensor with value normalization, similar to [0].

    This implementation is required since `torchvision.transforms.v2.ToTensor` is
    deprecated and will be removed in the future (see [1]).

    Input to this transform:
        PIL Image (H x W x C) of uint8 type in range [0,255]

    Output of this transform:
        torch.Tensor (C x H x W) of type torch.float32 in range [0.0, 1.0]

    - [0] https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    - [1] https://pytorch.org/vision/0.20/generated/torchvision.transforms.v2.ToTensor.html?highlight=totensor#torchvision.transforms.v2.ToTensor
    """
    T = torchvision_transforms
    if _TRANSFORMS_V2 and hasattr(T, "ToImage") and hasattr(T, "ToDtype"):
        # v2.transforms.ToTensor is deprecated and will be removed in the future.
        # This is the new recommended way to convert a PIL Image to a tensor since
        # torchvision v0.16.
        # See also https://github.com/pytorch/vision/blame/33e47d88265b2d57c2644aad1425be4fccd64605/torchvision/transforms/v2/_deprecated.py#L19
        return T.Compose([T.ToImage(), T.ToDtype(dtype=torch.float32, scale=True)])
    else:
        return T.ToTensor()
