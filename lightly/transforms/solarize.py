# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Union

import numpy as np
from PIL.Image import Image as PILImage
from torch import Tensor

from lightly.transforms.torchvision_v2_compatibility import functional as F


class RandomSolarization(object):
    """Implementation of random image Solarization.

    Utilizes torchvision's functional `solarize` transform. Solarization
    inverts all pixel values above or equal to a threshold (default: 128).
    When applied to floating-point tensors in `[0, 1]`, integer thresholds or
    thresholds greater than `1.0` are scaled by `1 / 255`.

    Attributes:
        prob:
            Probability to apply the transformation.
        threshold:
            Threshold for solarization (in `[0, 255]` for PIL/uint8 images, or
            in `[0.0, 1.0]` when passed as a float for floating-point tensors).
    """

    def __init__(self, prob: float = 0.5, threshold: Union[int, float] = 128):
        self.prob = prob
        self.threshold = threshold

    def __call__(self, sample: Union[Tensor, PILImage]) -> Union[Tensor, PILImage]:
        """Solarizes the given input image or tensor.

        Args:
            sample:
                PIL image or tensor to which solarize will be applied.

        Returns:
            Solarized image/tensor or original input.

        """
        prob = np.random.random_sample()
        if prob < self.prob:
            threshold: Union[int, float] = self.threshold
            if isinstance(sample, Tensor) and sample.is_floating_point():
                if isinstance(threshold, int) or threshold > 1.0:
                    threshold = threshold / 255.0
            solarized: Union[Tensor, PILImage] = F.solarize(sample, threshold=threshold)
            return solarized
        return sample
