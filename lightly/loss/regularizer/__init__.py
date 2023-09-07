"""The lightly.loss.regularizer package provides regularizers for self-supervised learning. """


# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    from lightly.loss.regularizer.co2 import CO2Regularizer
