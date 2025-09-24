from typing import Dict, List, Optional, Tuple

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE


class WMSETransform(MultiViewTransform):
    """Implements the transformations for W-MSE [0].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length num_samples.

    Applies the following augmentations by default:
        - Color jitter
        - Random gray scale
        - Random resized crop
        - Random horizontal flip
        - Random gaussian blur
        - ImageNet normalization

    - [0] Whitening for Self-Supervised Representation Learning, 2021, https://arxiv.org/pdf/2007.06346.pdf

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of tensors of length k.

    Attributes:
        num_samples:
            Number of views. Must be the same as num_samples in the WMSELoss.
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
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
            Probability that random gray scale is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.
    """

    def __init__(
        self,
        num_samples: int = 2,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.4,
        cj_hue: float = 0.1,
        min_scale: float = 0.2,
        random_gray_scale: float = 0.1,
        hf_prob: float = 0.5,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[int] = None,
        sigmas: Tuple[float, float] = (0.1, 2.0),
        normalize: Dict[str, List[float]] = IMAGENET_NORMALIZE,
    ):
        if num_samples < 1:
            raise ValueError("num_samples must be greater than or equal to 1")
        transform = T.Compose(
            [
                T.RandomApply(
                    [T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)], p=cj_prob
                ),
                T.RandomGrayscale(p=random_gray_scale),
                T.RandomResizedCrop(
                    input_size,
                    scale=(min_scale, 1.0),
                    interpolation=3,
                ),
                T.RandomHorizontalFlip(p=hf_prob),
                GaussianBlur(
                    kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur
                ),
                T.ToTensor(),
                T.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )
        super().__init__(transforms=[transform] * num_samples)
