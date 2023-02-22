from torch import Tensor
from PIL.Image import Image
from typing import List, Tuple, Union
import torchvision.transforms as T


class ImageGridTransform:
    """Transforms an image into multiple views.

    Args:
        transforms:
            A sequence of transforms. Every transform creates a new view and its
            correspondent local grid. Used for VICRegL. This argument in particular is
            a list of tuples comprehending the cropping and the corrisponding grid
            transformation and the non geometrical composition of transforms.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Union[Tensor, Image]) -> Union[List[Tensor], List[Image]]:
        """Transforms an image into multiple views.

        Every transform in self.transforms creates a new view.

        Args:
            image:
                Image to be transformed into multiple views and grids.

        Returns:
            List of views and grids tensors or PIL images. In the VICRegL implementation
            it has size:
            [
                [3, global_crop_size, global_crop_size],
                [3, local_crop_size, local_crop_size],
                [global_grid_size, global_grid_size, 2],
                [local_grid_size, local_grid_size, 2]
            ]

        """
        views, grids = [], []
        for transform in self.transforms:
            view, grid = transform[0].forward(image)
            views.append(transform[1](view))
            grids.append(grid)
        views += grids
        return views
