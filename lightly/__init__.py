"""Lightly is a computer vision framework for self-supervised learning.

With Lightly you can train deep learning models using
self-supervision. This means, that you don't require
any labels to train a model. Lightly has been built
to help you understand and work with large unlabeled datasets.
It is built on top of PyTorch and therefore fully compatible
with other frameworks such as Fast.ai.

The framework is structured into the following modules:

- **cli**:

  The lightly.cli module provides a command-line interface for training
  self-supervised models and creating embeddings.

- **core**:

  The lightly.core module offers one-liners for simple self-supervised learning.

- **data**:

  The lightly.data module provides a dataset wrapper and collate functions. The
  collate functions are in charge of the data augmentations which are crucial for
  self-supervised learning.

- **loss**:

  The lightly.loss module contains implementations of popular self-supervised training
  loss functions.

- **models**:

  The lightly.models module holds the implementation of the ResNet as well as heads
  for self-supervised methods. It currently implements the heads of:

  - Barlow Twins

  - BYOL

  - MoCo

  - NNCLR

  - SimCLR

  - SimSiam

  - SwaV

- **transforms**:

  The lightly.transforms module implements custom data transforms. Currently implements:

  - Gaussian Blur

  - Random Rotation

  - Random Solarization

- **utils**:

  The lightly.utils package provides global utility methods.
  The io module contains utility to save and load embeddings in a format which is
  understood by the Lightly library.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

__name__ = "lightly"
__version__ = "1.5.24"
