from torch import Tensor
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from typing import Tuple, Union
from PIL.Image import Image
import torchvision.transforms as T


class MAETransform(MultiViewTransform):
    """Implements the view augmentation for MAE [0].

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377

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
        input_size: Union[int, Tuple[int, int]] = 224,
        min_scale: float = 0.2,
        normalize: dict = IMAGENET_NORMALIZE,
    ):
        transform = MAEViewTransform(
            input_size=input_size,
            min_scale=min_scale,
            normalize=normalize,
        )
        super().__init__([transform])


class MAEViewTransform:
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 224,
        min_scale: float = 0.2,
        normalize: dict = IMAGENET_NORMALIZE,
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

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            Image (Tensor): The input image to apply the transforms to.

        Returns:
            Image (Tensor): The transformed image.

        """
        return self.transform(image)
