from typing import Any, List, Sequence

from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T


class MultiViewTransformV2:
    """Transforms an image into multiple views and is compatible with transforms v2.

    Args:
        transforms:
            A sequence of v2 transforms. Every transform creates a new view.

    """

    def __init__(self, transforms: Sequence[T.Compose]):
        self.transforms = transforms

    def __call__(self, *args: Any) -> List[Any]:
        """Transforms a data structure containing images, bounding boxes and masks
        into a sequence of multiple views.

        Every transform in self.transforms creates a new view.

        Args:
            *args:
                Arbitary positional arguments consisting of arbitrary data structures
                containing images, bounding boxes and masks.

        Returns:
            A list of views, where each view is a transformed version of *args.

        """
        return [transform(*args) for transform in self.transforms]
