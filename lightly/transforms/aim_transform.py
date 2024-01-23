from typing import Dict, List, Tuple, Union

from lightly.transforms.mae_transform import MAETransform
from lightly.transforms.utils import IMAGENET_NORMALIZE


class AIMTransform(MAETransform):
    """Implements the view augmentation for AIM [0].

    Uses the same parameters as MAE [1] but with larger min_scale.

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 1.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip

    - [0]: AIM, 2024, https://arxiv.org/abs/2401.08541
    - [1]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377

    Attributes:
        input_size:
            Size of the input image in pixels.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 224,
        min_scale: float = 0.4,
        normalize: Dict[str, List[float]] = IMAGENET_NORMALIZE,
    ):
        super().__init__(
            input_size=input_size,
            min_scale=min_scale,
            normalize=normalize,
        )
