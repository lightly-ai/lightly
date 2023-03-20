from typing import Optional, Tuple, Union

import torchvision.transforms as T
from PIL.Image import Image
from torch import Tensor

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.solarize import RandomSolarization
from lightly.transforms.utils import IMAGENET_NORMALIZE


class SMoGTransform(MultiViewTransform):
    """Implements the transformations for SMoG.

    Attributes:
        crop_sizes:
            Size of the input image in pixels for each crop category.
        crop_counts:
            Number of crops for each crop category.
        crop_min_scales:
            Min scales for each crop category.
        crop_max_scales:
            Max_scales for each crop category.
        gaussian_blur_probs:
            Probability of Gaussian blur for each crop category.
        gaussian_blur_kernel_sizes:
            Deprecated values in favour of sigmas.
        gaussian_blur_sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
        solarize_probs:
            Probability of solarization for each crop category.
        hf_prob:
            Probability that horizontal flip is applied.
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
        random_gray_scale:
            Probability of conversion to grayscale.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        crop_sizes: Tuple[int, int] = (224, 96),
        crop_counts: Tuple[int, int] = (4, 4),
        crop_min_scales: Tuple[float, float] = (0.2, 0.05),
        crop_max_scales: Tuple[float, float] = (1.0, 0.2),
        gaussian_blur_probs: Tuple[float, float] = (0.5, 0.1),
        gaussian_blur_kernel_sizes: Tuple[Optional[float], Optional[float]] = (
            None,
            None,
        ),
        gaussian_blur_sigmas: Tuple[float, float] = (0.1, 2),
        solarize_probs: Tuple[float, float] = (0.0, 0.2),
        hf_prob: float = 0.5,
        cj_prob: float = 1.0,
        cj_strength: float = 0.5,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.4,
        cj_hue: float = 0.2,
        random_gray_scale: float = 0.2,
        normalize: Union[None, dict] = IMAGENET_NORMALIZE,
    ):
        transforms = []
        for i in range(len(crop_sizes)):
            transforms.extend(
                [
                    SmoGViewTransform(
                        crop_size=crop_sizes[i],
                        crop_min_scale=crop_min_scales[i],
                        crop_max_scale=crop_max_scales[i],
                        gaussian_blur_prob=gaussian_blur_probs[i],
                        kernel_size=gaussian_blur_kernel_sizes[i],
                        sigmas=gaussian_blur_sigmas,
                        solarize_prob=solarize_probs[i],
                        hf_prob=hf_prob,
                        cj_prob=cj_prob,
                        cj_strength=cj_strength,
                        cj_bright=cj_bright,
                        cj_contrast=cj_contrast,
                        cj_sat=cj_sat,
                        cj_hue=cj_hue,
                        random_gray_scale=random_gray_scale,
                        normalize=normalize,
                    )
                ]
                * crop_counts[i]
            )

        super().__init__(transforms)


class SmoGViewTransform:
    def __init__(
        self,
        crop_size: int = 224,
        crop_min_scale: float = 0.2,
        crop_max_scale: float = 1.0,
        gaussian_blur_prob: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        solarize_prob: float = 0.0,
        hf_prob: float = 0.5,
        cj_prob: float = 1.0,
        cj_strength: float = 0.5,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.4,
        cj_hue: float = 0.2,
        random_gray_scale: float = 0.2,
        normalize: Union[None, dict] = IMAGENET_NORMALIZE,
    ):
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            T.RandomResizedCrop(crop_size, scale=(crop_min_scale, crop_max_scale)),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(
                kernel_size=kernel_size,
                prob=gaussian_blur_prob,
                sigmas=sigmas,
            ),
            RandomSolarization(prob=solarize_prob),
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        return self.transform(image)
