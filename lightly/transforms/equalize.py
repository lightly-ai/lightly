# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

from PIL import ImageOps
from PIL.Image import Image as PILImage


class HistogramEqualization(object):
    """Implementation of histogram equalization.

    Utilizes the integrated image operation `equalize` from Pillow. Histogram
    equalization redistributes the pixel intensities to enhance contrast.
    Example paper: https://arxiv.org/abs/2101.04909

    Attributes:
        mask:
            An optional mask. If provided, only the pixels selected by
            the mask are included in the operation.
    """

    def __init__(self, mask: PILImage = None):
        self.mask = mask

    def __call__(self, sample: PILImage) -> PILImage:
        """Equalize the input image histogram

        Args:
            sample:
                PIL image to equalize.

        Returns:
            Equalized image with uniform distribution of grayscale values.

        """

        return ImageOps.equalize(sample, self.mask)
