from typing import Dict, List, Optional, Tuple, Union

from lightly.transforms.byol_transform import BYOLView1Transform
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE


class MMCRTransform(MultiViewTransform):
    """Implements the transformations for MMCR[0], which
    are based on BYOL[1].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length k.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - Solarization
        - ImageNet normalization

    Please refer to the BYOL implementation for additional details.

    - [0]: Efficient Coding of Natural Images using Maximum Manifold Capacity
            Representations, 2023, https://arxiv.org/pdf/2303.03307.pdf
    - [1]: Bootstrap Your Own Latent, 2020, https://arxiv.org/pdf/2006.07733.pdf


    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of tensors of length k.

    Attributes:
        k: Number of views.
        transform: The transform to apply to each view.
    """

    def __init__(
        self,
        k: int = 8,
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
        if k < 1:
            raise ValueError("k must be greater than or equal to 1")
        transform = BYOLView1Transform(
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
            solarization_prob=solarization_prob,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            normalize=normalize,
        )
        super().__init__(transforms=[transform] * k)
