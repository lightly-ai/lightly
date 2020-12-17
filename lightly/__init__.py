"""Lightly is a computer vision framework for self-supervised learning.

With Lightly you can train deep learning models using
self-supervision. This means, that you don't require
any labels to train a model. Lightly has been built
to help you understand and work with large unlabeled datasets.
It is built on top of PyTorch and therefore fully compatible 
with other frameworks such as Fast.ai.

The framework is structured into the following modules:

- **api**: 

  The lightly.api module handles communication with the Lightly web-app.

- **cli**:

  The lightly.cli module provides a command-line interface for training 
  self-supervised models and embedding images. Furthermore, the command-line
  tool can be used to upload and download images from/to the Lightly web-app.

- **core**:

  The lightly.core module offers one-liners for simple self-supervised learning.

- **data**:

  The lightly.data module provides a dataset wrapper and collate functions. The
  collate functions are in charge of the data augmentations which are crucial for
  self-supervised learning.

- **embedding**:

  The lightly.embedding module combines the self-supervised models with a dataloader,
  optimizer, and loss function to provide a simple pytorch-lightning trainable.

- **loss**:

  The lightly.loss module contains implementations of popular self-supervised training
  loss functions.

- **models**:

  The lightly.models module holds the implementation of the ResNet as well as self-
  supervised methods. Currently implements:

  - SimCLR

  - MoCo

- **transforms**:

  The lightly.transforms module implements custom data transforms. Currently implements:

  - Gaussian Blur

  - Random Rotation

- **utils**:

  The lightly.utils package provides global utility methods.
  The io module contains utility to save and load embeddings in a format which is
  understood by the Lightly library.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

__name__ = 'lightly'
__version__ = '1.0.7'


try:
    # See (https://github.com/PyTorchLightning/pytorch-lightning)
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    __LIGHTLY_SETUP__
except NameError:
    __LIGHTLY_SETUP__ = False


if __LIGHTLY_SETUP__:
    # setting up lightly
    msg = f'Partial import of {__name__}=={__version__} during build process.' 
    print(msg)
else:
    # see if prefetch_generator is available
    try:
        import prefetch_generator
    except ImportError:
        _prefetch_generator_available = False
    else:
        _prefetch_generator_available = True

    def is_prefetch_generator_available():
        return _prefetch_generator_available

    # import core functionalities
    from lightly.core import train_model_and_embed_images
    from lightly.core import train_embedding_model
    from lightly.core import embed_images


    # compare current version v0 to other version v1
    def version_compare(v0, v1):
        v0 = [int(n) for n in v0.split('.')][::-1]
        v1 = [int(n) for n in v1.split('.')][::-1]
        pairs = list(zip(v0, v1))[::-1]
        for x, y in pairs:
            if x < y:
                return -1
            if x > y:
                return 1
        return 0


    # message if current version is not latest version
    def pretty_print_latest_version(latest_version, width=70):
        lines = [
            'There is a newer version of the package available.',
            'For compatability reasons, please upgrade your current version.',
            '> pip install lightly=={}'.format(latest_version),
        ]
        print('-' * width)
        for line in lines:
            print('| ' + line + (width - len(line) - 3) * " " + "|")
        print('-' * width)


    # check for latest version
    from lightly.api import get_version
    latest_version = get_version(__version__)
    if latest_version is not None:
        if version_compare(__version__, latest_version) < 0:
            # local version is behind latest version
            # pretty_print_latest_version(latest_version)
            pass
