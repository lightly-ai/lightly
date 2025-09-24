from typing import Any, Dict, List, Optional, Tuple, Union

from lightly.transforms.add_grid_transform import AddGridTransform
from lightly.transforms.multi_view_transform_v2 import MultiViewTransformV2
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE


class DetConSTransform(MultiViewTransformV2):
    """Implements the transformations for DetConS [0], based on SimCLR.[1]

    This transform creates two views of the input data, where the second view is different
    in that it does not apply Gaussian blurring.

    Input to this transform:
        Arbitrary data structure containing images and masks, that is compatible with
        torchvision transforms v2.[2]

    Output of this transform:
        A list of two views, where each view is a transformed version of the input.

    Applies the following augmentations by default:
        - RandomResizedCrop
        - RandomRotation
        - RandomHorizontalFlip
        - RandomVerticalFlip
        - ColorJitter
        - RandomGrayscale
        - GaussianBlur (only for the first view)

    Can additionally apply a segmentation of the image into a regular grid if not provided
    with a pre-segmented image.

    References:
        - [0] DetCon, 2021, https://arxiv.org/abs/2103.10957
        - [1] SimCLR, 2020, https://arxiv.org/abs/2002.05709
        - [2] torchvision Getting started with transforms v2, https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html

    Attributes:
        grid_size: Size of the grid segmentation as a tuple (num_rows, num_cols), or None
            if the segmentation mask is to be provided by the user.
        gaussian_blur_t1:
            Probability of applying Gaussian blur to the first view.
        gaussian_blur_t2:
            Probability of applying Gaussian blur to the second view.
        input_size:
            Size of the desired model input in pixels.
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
            Minimum size of the randomized crop relative to the input image.
        random_gray_scale:
            Probability of conversion to grayscale.
        kernel_size:
            Size of the Gaussian kernel for Gaussian blur.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from. If degrees is a number instead of sequence
            like (min, max), the range of degrees will be (-degrees, +degrees). The rotation
            is applied counter-clockwise.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.
    """

    def __init__(
        self,
        grid_size: Optional[Tuple[int, int]] = None,
        gaussian_blur_t1: float = 1.0,
        gaussian_blur_t2: float = 0.0,
        input_size: Union[Tuple[int, int], int] = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        kernel_size: Tuple[float, float] = (23, 23),
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Union[float, Tuple[float, float]] = 0.0,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ) -> None:
        self.grid_size = grid_size

        tr1: List[Union[AddGridTransform, DetConSViewTransform]] = []
        tr2: List[Union[AddGridTransform, DetConSViewTransform]] = []

        if self.grid_size is not None:
            grid_tr1 = AddGridTransform(
                num_rows=self.grid_size[0], num_cols=self.grid_size[1]
            )
            tr1 += [grid_tr1]
            grid_tr2 = AddGridTransform(
                num_rows=self.grid_size[0], num_cols=self.grid_size[1]
            )
            tr2 += [grid_tr2]

        tr1 += [
            DetConSViewTransform(
                gaussian_blur=gaussian_blur_t1,
                input_size=input_size,
                cj_prob=cj_prob,
                cj_strength=cj_strength,
                cj_bright=cj_bright,
                cj_contrast=cj_contrast,
                cj_sat=cj_sat,
                cj_hue=cj_hue,
                min_scale=min_scale,
                random_gray_scale=random_gray_scale,
                kernel_size=kernel_size,
                sigmas=sigmas,
                vf_prob=vf_prob,
                hf_prob=hf_prob,
                rr_prob=rr_prob,
                rr_degrees=rr_degrees,
                normalize=normalize,
            )
        ]
        tr2 += [
            DetConSViewTransform(
                gaussian_blur=gaussian_blur_t2,
                input_size=input_size,
                cj_prob=cj_prob,
                cj_strength=cj_strength,
                cj_bright=cj_bright,
                cj_contrast=cj_contrast,
                cj_sat=cj_sat,
                cj_hue=cj_hue,
                min_scale=min_scale,
                random_gray_scale=random_gray_scale,
                kernel_size=kernel_size,
                sigmas=sigmas,
                vf_prob=vf_prob,
                hf_prob=hf_prob,
                rr_prob=rr_prob,
                rr_degrees=rr_degrees,
                normalize=normalize,
            )
        ]

        super().__init__(transforms=[T.Compose(tr1), T.Compose(tr2)])


class DetConSViewTransform:
    def __init__(
        self,
        input_size: Union[Tuple[int, int], int] = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 1.0,
        kernel_size: Tuple[float, float] = (23, 23),
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Union[float, Tuple[float, float]] = 0.0,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ) -> None:
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            T.RandomApply([T.RandomRotation(rr_degrees)], p=rr_prob),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=kernel_size, sigma=sigmas)], p=gaussian_blur
            ),
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.tr = T.Compose(transform)

    def __call__(self, *args: Any) -> Any:
        """Applies the transforms to arbitrary positional arguments containing arbitrary
        data structures of images, bounding boxes and masks.

        Args:
            *args: Arbitrary positional arguments consisting of arbitrary data structures
                containing images, bounding boxes and masks.

        Returns:
            The transformed input in the same data structure as the input.
        """
        return self.tr(*args)
