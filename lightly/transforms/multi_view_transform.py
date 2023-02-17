from torchvision.transforms import Compose
from torch import Tensor
from typing import List


class MultiViewTransform:
    """Applies multiple transforms to an image and returns a list of transformed images.

    Args:
        transforms (Compose): A composition of PyTorch transforms.

    """

    def __init__(self, transforms: Compose):
        self.transforms = transforms

    def __call__(self, image: Tensor) -> List[Tensor]:
        """
        Applies the transforms to the input image.

        Args:
            Image (Tensor): The input image to apply the transforms to.

        Returns:
            List[Tensor]: A list of transformed images.

        """
        return [transform(image) for transform in self.transforms]
