from __future__ import annotations

from PIL.Image import Image
from torch import Tensor

from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE


class CAPITransform:
    """Implements the view augmentation for CAPI [0].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 1.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip

    - [0]: CAPI: Cluster and Predict Latent Patches for Improved Masked Image Modeling, 2025, https://arxiv.org/abs/2502.08769

    Attributes:
        input_size:
            Size of the input image in pixels.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        input_size: int | tuple[int, int] = 224,
        min_scale: float = 0.6,
        normalize: dict[str, list[float]] = IMAGENET_NORMALIZE,
    ):
        transforms = [
            T.RandomResizedCrop(
                input_size, scale=(min_scale, 1.0), interpolation=3
            ),  # 3 is bicubic
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
        if normalize:
            transforms.append(T.Normalize(mean=normalize["mean"], std=normalize["std"]))

        self.transform = T.Compose(transforms)

    def __call__(self, image: Tensor | Image) -> list[Tensor]:
        """Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        return [self.transform(image)]
