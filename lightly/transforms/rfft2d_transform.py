from typing import Union

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor


class RFFT2DTransform:
    """2D Fast Fourier Transform (RFFT2D) Transformation.

    This transformation applies the 2D Fast Fourier Transform (RFFT2D)
    to an image, converting it from the spatial domain to the frequency domain.

    Input:
        - Tensor of shape (C, H, W), where C is the number of channels.
        - PIL.Image can also be passed and will be converted to a Tensor internally.

    Output:
        - Tensor of shape (C, H, W) in the frequency domain, where C is the number of channels.
    """

    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __call__(self, image: Union[Tensor, Image.Image]) -> torch.Tensor:
        """Applies the 2D Fast Fourier Transform (RFFT2D) to the input image.

        Args:
            image (Union[Tensor, PIL.Image.Image]):
                Input image in either PIL.Image format or as a Tensor of shape (C, H, W).

        Returns:
            Tensor: The image in the frequency domain after applying RFFT2D, of shape (C, H, W).
        """

        if isinstance(image, Image.Image):
            image = self.to_tensor(image)

        rfft_image = torch.fft.rfft2(image)
        return rfft_image
