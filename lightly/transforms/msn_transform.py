from typing import Optional, Tuple, Union

import torchvision.transforms as T
from PIL.Image import Image
from torch import Tensor

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE


class MSNTransform(MultiViewTransform):
    """Implements the transformations for MSN [0].

    Generates a set of random and focal views for each input image. The generated output
    is (views, target, filenames) where views is list with the following entries:
    [random_views_0, random_views_1, ..., focal_views_0, focal_views_1, ...].

    - [0]: Masked Siamese Networks, 2022: https://arxiv.org/abs/2204.07141

    Attributes:
        random_size:
            Size of the random image views in pixels.
        focal_size:
            Size of the focal image views in pixels.
        random_views:
            Number of random views to generate.
        focal_views:
            Number of focal views to generate.
        random_crop_scale:
            Minimum and maximum size of the randomized crops for the relative to random_size.
        focal_crop_scale:
            Minimum and maximum size of the randomized crops relative to focal_size.
        cj_prob:
            Probability that color jittering is applied.
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
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        random_gray_scale:
            Probability of conversion to grayscale.
        hf_prob:
            Probability that horizontal flip is applied.
        vf_prob:
            Probability that vertical flip is applied.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.
    """

    def __init__(
        self,
        random_size: int = 224,
        focal_size: int = 96,
        random_views: int = 2,
        focal_views: int = 10,
        random_crop_scale: Tuple[float, float] = (0.3, 1.0),
        focal_crop_scale: Tuple[float, float] = (0.05, 0.3),
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        random_gray_scale: float = 0.2,
        hf_prob: float = 0.5,
        vf_prob: float = 0.0,
        normalize: dict = IMAGENET_NORMALIZE,
    ):
        random_view_transform = MSNViewTransform(
            crop_size=random_size,
            crop_scale=random_crop_scale,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            random_gray_scale=random_gray_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            normalize=normalize,
        )
        focal_view_transform = MSNViewTransform(
            crop_size=focal_size,
            crop_scale=focal_crop_scale,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            random_gray_scale=random_gray_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            normalize=normalize,
        )
        transforms = [random_view_transform] * random_views
        transforms += [focal_view_transform] * focal_views
        super().__init__(transforms=transforms)


class MSNViewTransform:
    def __init__(
        self,
        crop_size: int = 224,
        crop_scale: Tuple[float, float] = (0.3, 1.0),
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        random_gray_scale: float = 0.2,
        hf_prob: float = 0.5,
        vf_prob: float = 0.0,
        normalize: dict = IMAGENET_NORMALIZE,
    ):
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )
        transform = [
            T.RandomResizedCrop(size=crop_size, scale=crop_scale),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            T.ToTensor(),
            T.Normalize(mean=normalize["mean"], std=normalize["std"]),
        ]

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
