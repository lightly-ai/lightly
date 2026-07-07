.. _pixio:

PIXIO
=====

Example implementation of the PIXIO method. PIXIO builds on the `Masked Autoencoder
(MAE) <https://arxiv.org/abs/2111.06377>`_ and adapts it for dense prediction through
three changes: a much deeper decoder (32 blocks) that takes over pixel-level detail
modeling, a larger masking granularity that masks whole blocks of patches on a regular
grid instead of individual patches, and multiple class tokens whose mean is used as the
global image representation.

Key Components
--------------

- **Data Augmentations**: Like MAE, PIXIO relies only on random resized cropping.
- **Masking**: PIXIO masks 75% of the patches, but at a coarser granularity — whole
  ``grid_size`` x ``grid_size`` blocks of patches are masked together (4x4 by default),
  which prevents trivial reconstruction from neighboring patches.
- **Backbone**: A standard ViT with multiple class tokens (8 by default, realized via
  ``reg_tokens``).
- **Decoder**: A deep (32-block) decoder that reconstructs the masked pixels.
- **Reconstruction Loss**: A Mean Squared Error (MSE) loss between the predicted and the
  normalized pixel values of the masked patches.

Good to Know
------------

- **Masking granularity**: The paper's headline configuration uses a 4x4 grid and 8
  class tokens. The dense-prediction-optimal ablation uses a 2x2 grid and 4 class
  tokens.
- **Input resolution**: The reference model is trained at 256x256 with patch size 16 so
  that the 16x16 patch grid divides evenly into 4x4 blocks.

Reference:
    `In Pursuit of Pixel Supervision for Visual Pre-training, 2025 <https://arxiv.org/abs/2512.15715>`_

.. note::

    PIXIO requires `TIMM <https://github.com/huggingface/pytorch-image-models>`_ to be
    installed

    .. code-block:: bash

        pip install "lightly[timm]"

.. tabs::
    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/pixio.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/pixio.py

        .. literalinclude:: ../../../examples/pytorch/pixio.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/pixio.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/pixio.py

        .. literalinclude:: ../../../examples/pytorch_lightning/pixio.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/pixio.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/pixio.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Distributed Sampling is used in the dataloader

        Distributed Sampling makes sure that each distributed process sees only
        a subset of the data.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/pixio.py
