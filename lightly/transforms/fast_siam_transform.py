from typing import Optional, Tuple, Union

from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.simsiam_transform import SimSiamViewTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE


class FastSiamTransform(MultiViewTransform):
    """Implements the transformations for FastSiam.

    Attributes:
        num_views:
            Number of views (num_views = K+1 where K is the number of target views).
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value. For datasets with small images,
            such as CIFAR, it is recommended to set `cj_strength` to 0.5.
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

    """

    def __init__(
        self,
        num_views: int = 4,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.4,
        cj_hue: float = 0.1,
        min_scale: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        normalize: Union[None, dict] = IMAGENET_NORMALIZE,
    ):
        transforms = [
            SimSiamViewTransform(
                input_size=input_size,
                cj_prob=cj_prob,
                cj_strength=cj_strength,
                cj_bright=cj_bright,
                cj_contrast=cj_contrast,
                cj_sat=cj_sat,
                cj_hue=cj_hue,
                min_scale=min_scale,
                random_gray_scale=random_gray_scale,
                gaussian_blur=gaussian_blur,
                kernel_size=kernel_size,
                sigmas=sigmas,
                vf_prob=vf_prob,
                hf_prob=hf_prob,
                rr_prob=rr_prob,
                rr_degrees=rr_degrees,
                normalize=normalize,
            )
            for _ in range(num_views)
        ]
        super().__init__(transforms=transforms)
