# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

from typing import List

import numpy as np
import torch
from PIL import Image as Image
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision import transforms as T


class Jigsaw(object):
    """Implementation of Jigsaw image augmentation, inspired from PyContrast library.

    Generates n_grid**2 random crops and returns a list.

    This augmentation is instrumental to PIRL.

    Attributes:
        n_grid:
            Side length of the meshgrid, sqrt of the number of crops.
        img_size:
            Size of image.
        crop_size:
            Size of crops.
        transform:
            Transformation to apply on each crop.

    Examples:
        >>> from lightly.transforms import Jigsaw
        >>>
        >>> jigsaw_crop = Jigsaw(n_grid=3, img_size=255, crop_size=64, transform=transforms.ToTensor())
        >>>
        >>> # img is a PIL image
        >>> crops = jigsaw_crops(img)
    """

    def __init__(
        self,
        n_grid: int = 3,
        img_size: int = 255,
        crop_size: int = 64,
        transform: T.Compose = T.ToTensor(),
    ):
        self.n_grid = n_grid
        self.img_size = img_size
        self.crop_size = crop_size
        self.grid_size = int(img_size / self.n_grid)
        self.side = self.grid_size - self.crop_size
        self.transform = transform

        yy, xx = np.meshgrid(np.arange(n_grid), np.arange(n_grid))
        self.yy = np.reshape(yy * self.grid_size, (n_grid * n_grid,))
        self.xx = np.reshape(xx * self.grid_size, (n_grid * n_grid,))

    def __call__(self, img: PILImage) -> Tensor:
        """Performs the Jigsaw augmentation
        Args:
            img:
                PIL image to perform Jigsaw augmentation on.

        Returns:
            Torch tensor with stacked crops.
        """
        r_x = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        r_y = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        img = np.asarray(img, np.uint8)
        crops: List[PILImage] = []
        for i in range(self.n_grid * self.n_grid):
            crops.append(
                img[
                    self.xx[i] + r_x[i] : self.xx[i] + r_x[i] + self.crop_size,
                    self.yy[i] + r_y[i] : self.yy[i] + r_y[i] + self.crop_size,
                    :,
                ]
            )
        crop_images = [Image.fromarray(crop) for crop in crops]
        crop_tensors: Tensor = torch.stack(
            [self.transform(crop) for crop in crop_images]
        )
        permutation: List[int] = np.random.permutation(self.n_grid**2).tolist()
        return crop_tensors[permutation]
