import torch
from torch import nn
from typing import List
from dataclasses import dataclass
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

imagenet_normalize = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

@dataclass
class Location:
    i: float 
    j: float
    h: float
    w: float
    H: float
    W: float
    flip: bool = False


@dataclass
class ImageAndLocation:
    image: Image.Image
    location: Location


@dataclass
class ImageTensorAndGrid:
    image: torch.Tensor
    grid: torch.Tensor


class RandomResizedCropWithLocation(T.RandomResizedCrop):
    """
    Do a random resized crop and return both the resulting image and the grid

    """

    def forward(self, img: ImageAndLocation) -> ImageAndLocation:
        i, j, h, w = self.get_params(img.image, self.scale, self.ratio)
        width, height = T.functional.get_image_size(img.image)
        location = Location(i=i, j=j, h=h, w=w, H=height, W=width)
        imageandlocation = ImageAndLocation(
            image=T.functional.resized_crop(
                img.image, i, j, h, w, self.size, self.interpolation
            ),
            location=location,
        )
        return imageandlocation
        


class RandomHorizontalFlipReturnsIfFlip(T.RandomHorizontalFlip):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions


    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def forward(self, img: ImageAndLocation) -> ImageAndLocation:
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        
        if torch.rand(1) < self.p:
            img.image = F.hflip(img.image)
            img.location.flip = True
        return img


class ReturnGrid(nn.Module):
    def __init__(
        self,
        N=7,
        crop_size: int = 224,
        crop_min_scale: float = 0.05,
        crop_max_scale: float = 0.2,
        hf_prob: float = 0.5,
        normalize: dict = imagenet_normalize,
    ):
        super().__init__()
        self.N = N
        self.crop_size = crop_size
        self.crop_min_scale = crop_min_scale
        self.crop_max_scale = crop_max_scale
        self.hf_prob = hf_prob
        self.normalize = normalize

    def forward(self, img: Image.Image) -> ImageTensorAndGrid:

        location = Location(i=0, j=0, h=0, w=0, H=img.height, W=img.width)
        img_and_location = ImageAndLocation(image=img, location=location)
        img_and_location = RandomResizedCropWithLocation(
            size=self.crop_size, scale=(self.crop_min_scale, self.crop_max_scale)
        ).forward(img_and_location)

        img_and_location = RandomHorizontalFlipReturnsIfFlip(self.hf_prob).forward(
            img_and_location
        )
        transform = T.ToTensor()
        image_tensor = transform(img_and_location.image)
        image_tensor = T.Normalize(
            mean=self.normalize["mean"], std=self.normalize["std"]
        ).forward(image_tensor)
        grid = self.location_to_NxN_grid(location=img_and_location.location)

        return ImageTensorAndGrid(image=image_tensor, grid=grid)

    def location_to_NxN_grid(self, location: Location) -> torch.Tensor:
        size_h_case = location.h / self.N
        size_w_case = location.w / self.N
        half_size_h_case = size_h_case / 2
        half_size_w_case = size_w_case / 2
        final_grid_x = torch.zeros(self.N, self.N)
        final_grid_y = torch.zeros(self.N, self.N)

        final_grid_x[0][0] = location.i + half_size_h_case
        final_grid_y[0][0] = location.j + half_size_w_case
        for k in range(1, self.N):
            final_grid_x[k][0] = final_grid_x[k - 1][0] + size_h_case
            final_grid_y[k][0] = final_grid_y[k - 1][0]
        for l in range(1, self.N):
            final_grid_x[0][l] = final_grid_x[0][l - 1]
            final_grid_y[0][l] = final_grid_y[0][l - 1] + size_w_case
        for k in range(1, self.N):
            for l in range(1, self.N):
                final_grid_x[k][l] = final_grid_x[k - 1][l] + size_h_case
                final_grid_y[k][l] = final_grid_y[k][l - 1] + size_w_case

        final_grid = torch.stack([final_grid_x, final_grid_y], dim=-1)
        if location.flip:
            # start_grid = final_grid.clone()
            for k in range(0, self.N):
                for l in range(0, self.N // 2):
                    swap = final_grid[k, l].clone()
                    final_grid[k, l] = final_grid[k, self.N - 1 - l]
                    final_grid[k, self.N - 1 - l] = swap

        return final_grid
