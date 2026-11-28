# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import math
from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image as PILImage
from PIL.Image import Image
from torch import Tensor

from lightly.transforms.torchvision_v2_compatibility import (
    _TRANSFORMS_V2,
)
from lightly.transforms.torchvision_v2_compatibility import (
    functional as F,
)


class GaussianBlur:
    """Implementation of random Gaussian blur.

    Utilizes torchvision's functional `gaussian_blur` to apply a Gaussian blur
    to the input image or tensor with a certain probability. The blur is further
    randomized by sampling uniformly the standard deviation (sigma) of the
    Gaussian kernel.

    Attributes:
        prob:
            Probability with which the blur is applied.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.

    """

    def __init__(
        self,
        *,
        prob: float = 0.5,
        sigmas: Tuple[float, float] = (0.2, 2),
    ):
        self.prob = prob
        self.sigmas = sigmas

    def __call__(self, sample: Union[Tensor, Image]) -> Union[Tensor, Image]:
        """Blurs the image with a given probability.

        Args:
            sample:
                PIL image or tensor to which blur will be applied.

        Returns:
            Blurred image or original image.
        """
        prob = np.random.random_sample()
        if prob >= self.prob:
            return sample

        # Choose randomized std for Gaussian filtering (preserving #1052 sigma semantics).
        sigma = float(np.random.uniform(self.sigmas[0], self.sigmas[1]))
        if sigma <= 0.0:
            return sample

        # Use an odd kernel covering +/- 3*sigma, clamped per axis so torchvision's
        # reflect padding (kernel_size // 2) remains strictly smaller than the image
        # dimension even for tiny inputs (e.g. 1x1).
        if isinstance(sample, Tensor):
            height, width = sample.shape[-2], sample.shape[-1]
        else:
            width, height = sample.size

        base_kernel = max(1, 2 * math.ceil(3.0 * sigma) + 1)
        kernel_w = min(base_kernel, 2 * width - 1)
        kernel_h = min(base_kernel, 2 * height - 1)

        # torchvision functional.gaussian_blur expects kernel_size as [kernel_x, kernel_y].
        if isinstance(sample, Tensor):
            # Legacy torchvision v1 gaussian_blur indexes img.shape[-3], so normalize
            # 2D grayscale tensors (H, W) to (1, H, W) to match previous to_pil/to_tensor shape.
            tensor_input = sample.unsqueeze(0) if sample.ndim == 2 else sample
            # Older PyTorch/torchvision releases (e.g. torch 1.10 / torchvision 0.11)
            # do not support Half reflection padding on CPU; promote to float32 and restore.
            needs_fp32 = (
                tensor_input.dtype == torch.float16
                and tensor_input.device.type == "cpu"
            )
            if needs_fp32:
                tensor_input = tensor_input.to(dtype=torch.float32)
            blurred_tensor: Tensor = F.gaussian_blur(
                tensor_input,
                kernel_size=[kernel_w, kernel_h],
                sigma=[sigma, sigma],
            )
            if needs_fp32:
                blurred_tensor = blurred_tensor.to(dtype=sample.dtype)
            return blurred_tensor

        if not _TRANSFORMS_V2 or sample.mode in ("RGBa", "La"):
            # Older torchvision v1 (e.g. 0.11) drops mode when reconstructing PIL images
            # (turning CMYK into RGBA), and to_pil_image rejects premultiplied-alpha
            # modes (RGBa, La) across torchvision versions. Reconstruct via Pillow directly.
            pil_tensor = F.pil_to_tensor(sample)
            blurred_pil_tensor = F.gaussian_blur(
                pil_tensor,
                kernel_size=[kernel_w, kernel_h],
                sigma=[sigma, sigma],
            )
            if blurred_pil_tensor.shape[0] == 1:
                arr = blurred_pil_tensor.squeeze(0).numpy()
            else:
                arr = blurred_pil_tensor.permute(1, 2, 0).contiguous().numpy()
            blurred_pil: Image = PILImage.fromarray(arr, mode=sample.mode)
            return blurred_pil

        blurred: Image = F.gaussian_blur(
            sample,
            kernel_size=[kernel_w, kernel_h],
            sigma=[sigma, sigma],
        )
        return blurred
