from torchvision.transforms import Compose
from torch import Tensor
from typing import List


class MultiViewTransform:
    """Transforms an image into multiple views.

    Args:
        transforms: 
            A sequence of transforms. Every transform creates a new view.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Tensor) -> List[Tensor]:
        """Transforms an image into multiple views.
        
        Every transform in self.transforms creates a new view.

        Args:
            image: 
                Image to be transformed into multiple views.

        Returns:
            List of views.

        """
        return [transform(image) for transform in self.transforms]
