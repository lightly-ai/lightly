from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)


class TiCoTransform(BYOLTransform):
    """Implements the transformations for TiCo[0]. These are the same as BYOL[1].

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

    Note that SimCLR v1 and v2 use similar augmentations. In detail, TiCo (and BYOL) has
    asymmetric gaussian blur and solarization. Furthermore, TiCo has weaker
    color jitter compared to SimCLR.

    - [0]: Jiachen Zhu et. al, 2022, Tico, https://arxiv.org/abs/2206.10698.pdf
    - [1]: Bootstrap Your Own Latent, 2020, https://arxiv.org/pdf/2006.07733.pdf

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of [tensor, tensor].

    Attributes:
        view_1_transform: The transform for the first view.
        view_2_transform: The transform for the second view.
    """


class TiCoView1Transform(BYOLView1Transform):
    """Alias for BYOLView1Transform."""


class TiCoView2Transform(BYOLView2Transform):
    """Alias for BYOLView2Transform."""
