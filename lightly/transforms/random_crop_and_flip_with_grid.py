import torch
from torch import nn
from typing import List, Tuple, Dict
from dataclasses import dataclass
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


@dataclass
class Location:
    # The row index of the top-left corner of the crop.
    top: float 
    # The column index of the top-left corner of the crop.
    left: float
    # The height of the crop.
    height: float
    # The width of the crop.
    width: float
    # The height of the original image.
    HEIGHT: float
    # The width of the original image.
    WIDTH: float
    # Whether to flip the image horizontally.
    horizontal_flip: bool = False
    # Whether to flip the image vertically.
    vertical_flip: bool = False


class RandomResizedCropWithLocation(T.RandomResizedCrop):
    """
    Do a random resized crop and return both the resulting image and the grid

    """

    def forward(self, img: Image.Image) -> Tuple[Image.Image, Location]:
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Randomly cropped image
            Location: Location object containing crop parameters
            
        """
        top, left, height, width = self.get_params(img, self.scale, self.ratio)
        width, height = T.functional.get_image_size(img)
        location = Location(top=top, left=left, height=height, width=width, HEIGHT=height, WIDTH=width)
        img = T.functional.resized_crop(
                img, top, left, height, width, self.size, self.interpolation, antialias=self.antialias
            )
        return img, location
        


class RandomHorizontalFlipWithLocation(T.RandomHorizontalFlip):
    """
    Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions


    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def forward(self, img: Image.Image, location:Location) -> Tuple[Image.Image, Location]:
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped..
            Location: Location object linked to the image
        Returns:
            PIL Image or Tensor: Randomly flipped image 
            Location: Location object with updated location.horizontal_flip parameter
        """
        
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            location.horizontal_flip = True
        return img, location

class RandomVerticalFlipWithLocation(T.RandomVerticalFlip):

    """Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions


    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """


    def forward(self, img: Image.Image, location:Location) -> Tuple[Image.Image, Location]:
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped..
            Location: Location object linked to the image
        Returns:
            PIL Image or Tensor: Randomly flipped image 
            Location: Location object with updated location.vertical_flip parameter

        """
        
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            location.vertical_flip = True
        return img, location

class RandomResizedCropAndFlip(nn.Module):
    """
        A PyTorch module that applies random cropping and horizontal flipping to an image,
        and returns the transformed image and a grid tensor used to map the image back to the
        original image space in an NxN grid.

        Args:
            N: 
                The number of grid cells in the output grid tensor.
            crop_size: 
                The size (in pixels) of the random crops.
            crop_min_scale: 
                The minimum scale factor for random resized crops.
            crop_max_scale:
                The maximum scale factor for random resized crops.
            hf_prob: 
                The probability of applying horizontal flipping to the image.
            normalize: 
                A dictionary containing the mean and std values for normalizing the image.

        """

    def __init__(
        self,
        grid_size: int = 7,
        crop_size: int = 224,
        crop_min_scale: float = 0.05,
        crop_max_scale: float = 0.2,
        hf_prob: float = 0.5,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.crop_size = crop_size
        self.crop_min_scale = crop_min_scale
        self.crop_max_scale = crop_max_scale
        self.hf_prob = hf_prob
        self.resized_crop = RandomResizedCropWithLocation(
            size=self.crop_size, scale=(self.crop_min_scale, self.crop_max_scale)
        )
        self.horizontal_flip = RandomHorizontalFlipWithLocation(self.hf_prob)
        self.vertical_flip = RandomVerticalFlipWithLocation(self.hf_prob)

    def forward(self, img: Image.Image) -> Tuple[Image.Image, Location]:

        """
        
        Applies random cropping and horizontal flipping to an image, and returns the
        transformed image and a grid tensor used to map the image back to the original image
        space in an NxN grid.

        Args:
            img: The input image.

        Returns:
            An `ImageTensorAndGrid` object containing the transformed image tensor and the
            grid tensor.

        """

        img, location = self.resized_crop.forward(img=img)
        img, location = self.horizontal_flip.forward(img, location)
        img, location = self.vertical_flip.forward(img, location)
        grid = self.location_to_NxN_grid(location=location)

        return img, grid

    def location_to_NxN_grid(self, location: Location) -> torch.Tensor:

        """
        Create a grid tensor with grid_size rows and grid_size columns, where each cell represents a region of
        the original image. The grid is used to map the cropped and transformed image back to the
        original image space.

        Args:
            location: An instance of the Location class, containing the location and size of the
                transformed image in the original image space.

        Returns:
            A grid tensor of shape (grid_size, grid_size, 2), where the last dimension represents the (x, y) coordinate
            of the center of each cell in the original image space.
        """
        
        cell_width = location.width / self.grid_size 
        cell_height = location.height / self.grid_size
        x = torch.linspace(location.left, location.left + location.width, self.grid_size) + (cell_width / 2)
        y = torch.linspace(location.top, location.top + location.height, self.grid_size) + (cell_height / 2)
        if location.horizontal_flip:
            x = torch.flip(x, dims=[0])
        if location.vertical_flip:
            y = torch.flip(y, dims=[0])
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        return torch.stack([grid_x, grid_y], dim=-1)
