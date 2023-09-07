"""The lightly.data module provides a dataset wrapper and collate functions. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    from lightly.data._video import (
        EmptyVideoError,
        NonIncreasingTimestampError,
        UnseekableTimestampError,
        VideoError,
    )
    from lightly.data.collate import (
        BaseCollateFunction,
        DINOCollateFunction,
        ImageCollateFunction,
        MAECollateFunction,
        MoCoCollateFunction,
        MSNCollateFunction,
        MultiCropCollateFunction,
        PIRLCollateFunction,
        SimCLRCollateFunction,
        SwaVCollateFunction,
        VICRegLCollateFunction,
        imagenet_normalize,
    )
    from lightly.data.dataset import LightlyDataset
