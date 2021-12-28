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

__name__ = 'lightly'
__version__ = '1.2.1'

import os

try:
    # See (https://github.com/PyTorchLightning/pytorch-lightning)
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    __LIGHTLY_SETUP__
except NameError:
    __LIGHTLY_SETUP__ = False


if __LIGHTLY_SETUP__:
    # setting up lightly
    msg = f'Partial import of {__name__}=={__version__} during build process.' 
    print(msg)
else:
    # see if prefetch_generator is available
    try:
        import prefetch_generator
    except ImportError:
        _prefetch_generator_available = False
    else:
        _prefetch_generator_available = True

    def _is_prefetch_generator_available():
        return _prefetch_generator_available

    from lightly.core import *
    from lightly import active_learning
    from lightly import api
    from lightly import data
    from lightly import embedding
    from lightly import loss
    from lightly import models
    from lightly import transforms
    from lightly import utils
    
    from lightly.api.version_checking import do_version_check

    if os.getenv('LIGHTLY_DID_VERSION_CHECK', 'False') == 'False':
        os.environ['LIGHTLY_DID_VERSION_CHECK'] = 'True'

        try:
            do_version_check(current_version=__version__)
        except Exception as e:
            pass



