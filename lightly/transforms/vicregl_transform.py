from typing import Optional, Tuple, Union

import torchvision.transforms as T
from PIL.Image import Image
from torch import Tensor

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.image_grid_transform import ImageGridTransform
from lightly.transforms.random_crop_and_flip_with_grid import RandomResizedCropAndFlip
from lightly.transforms.solarize import RandomSolarization
from lightly.transforms.utils import IMAGENET_NORMALIZE


class VICRegLTransform(ImageGridTransform):
    """Transforms images for VICRegL.

    - [0]: VICRegL, 2022, https://arxiv.org/abs/2210.01571

    Attributes:
        global_crop_size:
            Size of the input image in pixels for the global crop views.
        local_crop_size:
            Size of the input image in pixels for the local crop views.
        n_global_views:
            Number of global crop views to generate.
        n_local_views:
            Number of local crop views to generate. For ResNet backbones it is
            recommended to set this to 0, see [0].
        global_crop_scale:
            Min and max scales for the global crop views.
        local_crop_scale:
            Min and max scales for the local crop views.
        global_grid_size:
            Grid size for the global crop views.
        local_grid_size:
            Grid size for the local crop views.
        global_gaussian_blur_prob:
            Probability of Gaussian blur for the global crop views.
        local_gaussian_blur_prob:
            Probability of Gaussian blur for the local crop views.
        global_gaussian_blur_kernel_size:
            Will be deprecated in favor of `global_gaussian_blur_sigmas` argument.
            If set, the old behavior applies and `global_gaussian_blur_sigmas`
            is ignored. Used to calculate sigma of gaussian blur with
            global_gaussian_blur_kernel_size * input_size. Applied to global crop views.
        local_gaussian_blur_kernel_size:
            Will be deprecated in favor of `local_gaussian_blur_sigmas` argument.
            If set, the old behavior applies and `local_gaussian_blur_sigmas`
            is ignored. Used to calculate sigma of gaussian blur with
            local_gaussian_blur_kernel_size * input_size. Applied to local crop views.
        global_gaussian_blur_sigmas:
            Tuple of min and max value from which the std of the gaussian kernel
            is sampled. It is ignored if `global_gaussian_blur_kernel_size` is set.
            Applied to global crop views.
        local_gaussian_blur_sigmas:
            Tuple of min and max value from which the std of the gaussian kernel
            is sampled. It is ignored if `local_gaussian_blur_kernel_size` is set.
            Applied to local crop views.
        global_solarize_prob:
            Probability of solarization for the global crop views.
        local_solarize_prob:
            Probability of solarization for the local crop views.
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
            Dictionary with mean and standard deviation for normalization.
    """

    def __init__(
        self,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        n_global_views: int = 2,
        n_local_views: int = 6,
        global_crop_scale: Tuple[float, float] = (0.2, 1.0),
        local_crop_scale: Tuple[float, float] = (0.05, 0.2),
        global_grid_size: int = 7,
        local_grid_size: int = 3,
        global_gaussian_blur_prob: float = 0.5,
        local_gaussian_blur_prob: float = 0.1,
        global_gaussian_blur_kernel_size: Optional[float] = None,
        local_gaussian_blur_kernel_size: Optional[float] = None,
        global_gaussian_blur_sigmas: Tuple[float, float] = (0.1, 2),
        local_gaussian_blur_sigmas: Tuple[float, float] = (0.1, 2),
        global_solarize_prob: float = 0.0,
        local_solarize_prob: float = 0.2,
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
        global_transform = (
            RandomResizedCropAndFlip(
                crop_size=global_crop_size,
                crop_min_scale=global_crop_scale[0],
                crop_max_scale=global_crop_scale[1],
                hf_prob=hf_prob,
                grid_size=global_grid_size,
            ),
            VICRegLViewTransform(
                gaussian_blur_prob=global_gaussian_blur_prob,
                gaussian_blur_kernel_size=global_gaussian_blur_kernel_size,
                gaussian_blur_sigmas=global_gaussian_blur_sigmas,
                solarize_prob=global_solarize_prob,
                cj_prob=cj_prob,
                cj_strength=cj_strength,
                cj_bright=cj_bright,
                cj_contrast=cj_contrast,
                cj_sat=cj_sat,
                cj_hue=cj_hue,
                random_gray_scale=random_gray_scale,
                normalize=normalize,
            ),
        )
        local_transform = (
            RandomResizedCropAndFlip(
                crop_size=local_crop_size,
                crop_min_scale=local_crop_scale[0],
                crop_max_scale=local_crop_scale[1],
                hf_prob=hf_prob,
                grid_size=local_grid_size,
            ),
            VICRegLViewTransform(
                gaussian_blur_prob=local_gaussian_blur_prob,
                gaussian_blur_kernel_size=local_gaussian_blur_kernel_size,
                gaussian_blur_sigmas=local_gaussian_blur_sigmas,
                solarize_prob=local_solarize_prob,
                cj_prob=cj_prob,
                cj_strength=cj_strength,
                random_gray_scale=random_gray_scale,
                normalize=normalize,
            ),
        )

        transforms = [global_transform] * n_global_views + [
            local_transform
        ] * n_local_views
        super().__init__(transforms=transforms)


class VICRegLViewTransform:
    def __init__(
        self,
        gaussian_blur_prob: float = 0.5,
        gaussian_blur_kernel_size: Optional[float] = None,
        gaussian_blur_sigmas: Tuple[float, float] = (0.1, 2),
        solarize_prob: float = 0.0,
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

        transforms = [
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(
                kernel_size=gaussian_blur_kernel_size,
                prob=gaussian_blur_prob,
                sigmas=gaussian_blur_sigmas,
            ),
            RandomSolarization(prob=solarize_prob),
            T.ToTensor(),
        ]
        if normalize:
            transforms += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transforms=transforms)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.
        """
        return self.transform(image)
