from torch import Tensor
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.transforms.rotation import RandomRotationTransform
from lightly.transforms.gaussian_blur import GaussianBlur
from typing import Tuple, Union
import torchvision.transforms as T


class SimCLRTransform(MultiViewTransform):
    """Implements the transformations for SimCLR.

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    Examples:

        >>> # SimCLR for ImageNet
        >>> collate_fn = SimCLRCollateFunction()
        >>>
        >>> # SimCLR for CIFAR-10
        >>> collate_fn = SimCLRCollateFunction(
        >>>     input_size=32,
        >>>     gaussian_blur=0.,
        >>> )

    """

    def __init__(
        self,
        input_size: int = 64,
        cj_prob: float = 0.8,
        cj_bright: float = 0.7,
        cj_contrast: float = 0.7,
        cj_sat: float = 0.7,
        cj_hue: float = 0.2,
        min_scale: float = 0.15,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        sigmas: Tuple[float, float] = (0.2, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        normalize: Union[None, dict] = IMAGENET_NORMALIZE,
        to_tensor: bool = True,
    ):

        view_transform = SimCLRViewTransform(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            sigmas=sigmas,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            normalize=normalize,
            to_tensor=to_tensor,
        )
        super().__init__(transforms=[view_transform, view_transform])


class SimCLRViewTransform:
    def __init__(
        self,
        input_size: int = 64,
        cj_prob: float = 0.8,
        cj_bright: float = 0.7,
        cj_contrast: float = 0.7,
        cj_sat: float = 0.7,
        cj_hue: float = 0.2,
        min_scale: float = 0.15,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        sigmas: Tuple[float, float] = (0.2, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        normalize: Union[None, dict] = IMAGENET_NORMALIZE,
        to_tensor: bool = True,
    ):
        color_jitter = T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            RandomRotationTransform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(sigmas=sigmas, prob=gaussian_blur),
        ]
        if to_tensor:
            transform.append(T.ToTensor())
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Tensor) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            Image (Tensor): The input image to apply the transforms to.

        Returns:
            Image (Tensor): The transformed image.

        """
        return self.transform(image)
