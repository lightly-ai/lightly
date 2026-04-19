"""The lightly.embedding module provides trainable embedding strategies.

The embedding models use a pre-trained ResNet but should be finetuned on each
dataset instance.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.embedding._base import BaseEmbedding
from lightly.embedding.embedding import SelfSupervisedEmbedding
