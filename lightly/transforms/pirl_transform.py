from typing import Dict, List, Tuple, Union

from lightly.transforms.jigsaw import Jigsaw
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE


class PIRLTransform(MultiViewTransform):
    """Implements the transformations for PIRL [0]. The jigsaw augmentation
    is applied during the forward pass.

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2 (original, augmented).

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Jigsaw puzzle

    - [0] PIRL, 2019: https://arxiv.org/abs/1912.01991

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        hf_prob:
            Probability that horizontal flip is applied.
        n_grid:
            Sqrt of the number of grids in the jigsaw image.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 64,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.4,
        cj_hue: float = 0.4,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        hf_prob: float = 0.5,
        n_grid: int = 3,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        if isinstance(input_size, tuple):
            input_size_ = max(input_size)
        else:
            input_size_ = input_size

        # Cropping and normalisation for non-transformed image
        transforms_no_augment = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            T.ToTensor(),
        ]

        if normalize is not None:
            transforms_no_augment.append(
                T.Normalize(mean=normalize["mean"], std=normalize["std"])
            )

        no_augment = T.Compose(transforms_no_augment)

        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        # Transform for transformed jigsaw image
        transforms = [
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            T.ToTensor(),
        ]

        if normalize is not None:
            transforms.append(T.Normalize(mean=normalize["mean"], std=normalize["std"]))

        jigsaw = Jigsaw(
            n_grid=n_grid,
            img_size=input_size_,
            crop_size=int(input_size_ // n_grid),
            transform=T.Compose(transforms),
        )

        super().__init__([no_augment, jigsaw])
