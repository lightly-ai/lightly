from typing import Union

from byol_transform import BYOLView1Transform

from lightly.transforms.multi_view_transform import MultiViewTransform


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
        k: int = 2,
        transform: Union[BYOLView1Transform, None] = None,
    ):
        if k < 1:
            raise ValueError("k must be greater than or equal to 1")
        transform = transform or BYOLView1Transform()
        super().__init__(transforms=[transform] * k)
