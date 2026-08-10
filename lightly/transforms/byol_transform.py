from typing import Dict, List, Optional, Tuple, Union

from PIL.Image import Image
from torch import Tensor

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.solarize import RandomSolarization
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE


class BYOLView1Transform:
    """Transforms an image into the first view for BYOL [0].

    Used by BYOLTransform to create the first view of an image.

    Input to this transform:
        PIL Image. (Tensor inputs are supported when torchvision transforms v2 are available.)
    Output of this transform:
        Tensor.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - ImageNet normalization

    - [0]: Bootstrap Your Own Latent, 2020, https://arxiv.org/pdf/2006.07733.pdf

    """

    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.2,
        cj_hue: float = 0.1,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 1.0,
        solarization_prob: float = 0.0,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        """Initializes BYOLView1Transform.

        Args:
            input_size: Size of the input image in pixels.
            cj_prob: Probability that color jitter is applied.
            cj_strength: Strength of the color jitter. `cj_bright`, `cj_contrast`,
                `cj_sat`, and `cj_hue` are multiplied by this value. For datasets
                with small images, such as CIFAR, it is recommended to set
                `cj_strength` to 0.5.
            cj_bright: How much to jitter brightness.
            cj_contrast: How much to jitter contrast.
            cj_sat: How much to jitter saturation.
            cj_hue: How much to jitter hue.
            min_scale: Minimum size of the randomized crop relative to the input_size.
            random_gray_scale: Probability of conversion to grayscale.
            gaussian_blur: Probability of Gaussian blur.
            solarization_prob: Probability of solarization.
            kernel_size: Will be deprecated in favor of `sigmas` argument. If set,
                the old behavior applies and `sigmas` is ignored. Used to calculate
                sigma of gaussian blur with kernel_size * input_size.
            sigmas: Tuple of min and max value from which the std of the gaussian
                kernel is sampled. Is ignored if `kernel_size` is set.
            vf_prob: Probability that vertical flip is applied.
            hf_prob: Probability that horizontal flip is applied.
            rr_prob: Probability that random rotation is applied.
            rr_degrees: Range of degrees to select from for random rotation. If
                rr_degrees is None, images are rotated by 90 degrees. If rr_degrees
                is a (min, max) tuple, images are rotated by a random angle in
                [min, max]. If rr_degrees is a single number, images are rotated by
                a random angle in [-rr_degrees, +rr_degrees]. All rotations are
                counter-clockwise.
            normalize: Dictionary with 'mean' and 'std' for
                torchvision.transforms.Normalize.

        """
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            RandomSolarization(prob=solarization_prob),
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """Applies the transforms to the input image.

        Args:
            image: The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed


class BYOLView2Transform:
    """Transforms an image into the second view for BYOL [0].

    Used by BYOLTransform to create the second view of an image.

    Input to this transform:
        PIL Image. (Tensor inputs are supported when torchvision transforms v2 are available.)
    Output of this transform:
        Tensor.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - Solarization
        - ImageNet normalization

    - [0]: Bootstrap Your Own Latent, 2020, https://arxiv.org/pdf/2006.07733.pdf

    """

    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.2,
        cj_hue: float = 0.1,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.1,
        solarization_prob: float = 0.2,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        """Initializes BYOLView2Transform.

        Args:
            input_size: Size of the input image in pixels.
            cj_prob: Probability that color jitter is applied.
            cj_strength: Strength of the color jitter. `cj_bright`, `cj_contrast`,
                `cj_sat`, and `cj_hue` are multiplied by this value. For datasets
                with small images, such as CIFAR, it is recommended to set
                `cj_strength` to 0.5.
            cj_bright: How much to jitter brightness.
            cj_contrast: How much to jitter contrast.
            cj_sat: How much to jitter saturation.
            cj_hue: How much to jitter hue.
            min_scale: Minimum size of the randomized crop relative to the input_size.
            random_gray_scale: Probability of conversion to grayscale.
            gaussian_blur: Probability of Gaussian blur.
            solarization_prob: Probability of solarization.
            kernel_size: Will be deprecated in favor of `sigmas` argument. If set,
                the old behavior applies and `sigmas` is ignored. Used to calculate
                sigma of gaussian blur with kernel_size * input_size.
            sigmas: Tuple of min and max value from which the std of the gaussian
                kernel is sampled. Is ignored if `kernel_size` is set.
            vf_prob: Probability that vertical flip is applied.
            hf_prob: Probability that horizontal flip is applied.
            rr_prob: Probability that random rotation is applied.
            rr_degrees: Range of degrees to select from for random rotation. If
                rr_degrees is None, images are rotated by 90 degrees. If rr_degrees
                is a (min, max) tuple, images are rotated by a random angle in
                [min, max]. If rr_degrees is a single number, images are rotated by
                a random angle in [-rr_degrees, +rr_degrees]. All rotations are
                counter-clockwise.
            normalize: Dictionary with 'mean' and 'std' for
                torchvision.transforms.Normalize.

        """
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            RandomSolarization(prob=solarization_prob),
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """Applies the transforms to the input image.

        Args:
            image: The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed


class BYOLTransform(MultiViewTransform):
    """Implements the transformations for BYOL[0].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - Solarization
        - ImageNet normalization

    Note that SimCLR v1 and v2 use similar augmentations. In detail, BYOL has
    asymmetric gaussian blur and solarization. Furthermore, BYOL has weaker
    color jitter compared to SimCLR.

    - [0]: Bootstrap Your Own Latent, 2020, https://arxiv.org/pdf/2006.07733.pdf

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of [tensor, tensor].

    Attributes:
        view_1_transform: The transform for the first view.
        view_2_transform: The transform for the second view.
    """

    def __init__(
        self,
        view_1_transform: Optional[BYOLView1Transform] = None,
        view_2_transform: Optional[BYOLView2Transform] = None,
    ):
        # We need to initialize the transforms here
        view_1_transform = view_1_transform or BYOLView1Transform()
        view_2_transform = view_2_transform or BYOLView2Transform()
        super().__init__(transforms=[view_1_transform, view_2_transform])
