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

try:
    from torchvision.transforms import v2 as torchvision_transforms

    _TRANSFORMS_V2 = True

except ImportError:
    from torchvision import transforms as torchvision_transforms

    _TRANSFORMS_V2 = False
