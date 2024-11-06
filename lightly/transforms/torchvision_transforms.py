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

try:
    from torchvision.transforms import v2 as torchvision_transforms

    _TRANSFORMS_V2 = True

except ImportError:
    from torchvision import transforms as torchvision_transforms

    _TRANSFORMS_V2 = False


def ToTensor() -> Union[torchvision_transforms.Compose, ToTensorV1]:
    T = torchvision_transforms
    if _TRANSFORMS_V2 and hasattr(T, "ToImage") and hasattr(T, "ToDtype"):
        # v2.transforms.ToTensor is deprecated and will be removed in the future.
        # This is the new recommended way to convert a PIL Image to a tensor since
        # torchvision v0.16.
        # See also https://github.com/pytorch/vision/blame/33e47d88265b2d57c2644aad1425be4fccd64605/torchvision/transforms/v2/_deprecated.py#L19
        return T.Compose([T.ToImage(), T.ToDtype(dtype=torch.float32, scale=True)])
    else:
        return T.ToTensor()
