from typing import List, Sequence, Union

from PIL.Image import Image
from torch import Tensor

from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T


class ImageGridTransform:
    """Transforms an image into multiple views and grids.

    Used for VICRegL.

    Attributes:
        transforms:
            A sequence of (image_grid_transform, view_transform) tuples.
            The image_grid_transform creates a new view and grid from the image.
            The view_transform further augments the view. Every transform tuple
            is applied once to the image, creating len(transforms) views and
            grids.
    """

    def __init__(self, transforms: Sequence[T.Compose]):
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
        for image_grid_transform, view_transform in self.transforms:
            view, grid = image_grid_transform(image)
            views.append(view_transform(view))
            grids.append(grid)
        views += grids
        return views
