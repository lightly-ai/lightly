from typing import Tuple

import torchvision.transforms as T

from lightly.transforms.multi_view_transform import MultiViewTransform


class MultiCropTranform(MultiViewTransform):
    """Implements the multi-crop transformations. Used by Swav.

    Attributes:
        crop_sizes:
            Size of the input image in pixels for each crop category.
        crop_counts:
            Number of crops for each crop category.
        crop_min_scales:
            Min scales for each crop category.
        crop_max_scales:
            Max_scales for each crop category.
        transforms:
            Transforms which are applied to all crops.

    """

    def __init__(
        self,
        crop_sizes: Tuple[int],
        crop_counts: Tuple[int],
        crop_min_scales: Tuple[float],
        crop_max_scales: Tuple[float],
        transforms,
    ):
        if len(crop_sizes) != len(crop_counts):
            raise ValueError(
                "Length of crop_sizes and crop_counts must be equal but are"
                f" {len(crop_sizes)} and {len(crop_counts)}."
            )
        if len(crop_sizes) != len(crop_min_scales):
            raise ValueError(
                "Length of crop_sizes and crop_min_scales must be equal but are"
                f" {len(crop_sizes)} and {len(crop_min_scales)}."
            )
        if len(crop_sizes) != len(crop_min_scales):
            raise ValueError(
                "Length of crop_sizes and crop_max_scales must be equal but are"
                f" {len(crop_sizes)} and {len(crop_min_scales)}."
            )

        crop_transforms = []
        for i in range(len(crop_sizes)):
            random_resized_crop = T.RandomResizedCrop(
                crop_sizes[i], scale=(crop_min_scales[i], crop_max_scales[i])
            )

            crop_transforms.extend(
                [
                    T.Compose(
                        [
                            random_resized_crop,
                            transforms,
                        ]
                    )
                ]
                * crop_counts[i]
            )
        super().__init__(crop_transforms)
