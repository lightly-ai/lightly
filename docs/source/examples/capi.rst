.. _capi:

CAPI
====

Example implementation of the CAPI method. CAPI is a masked image modeling method
that trains a student to predict the *cluster assignments* of masked image patches
rather than their pixels. An exponential moving average teacher encodes the full
image and assigns each patch to one of many learned clusters; a positionwise
Sinkhorn-Knopp normalization turns these assignments into balanced soft targets. The
student sees only the visible patches and predicts the assignments of the masked
patches through a predictor that attends to the visible tokens.

Key Components
--------------

- **Data Augmentations**: CAPI relies only on random resized cropping and horizontal
  flipping.
- **Masking**: Inverse-block masking keeps a single contiguous block of patches
  visible and masks the rest, which are then predicted from the visible ones.
- **Backbone**: A standard ViT encodes the visible patches.
- **Predictor**: Mask-token queries attend to the visible encoder tokens to predict
  the masked patches.
- **Clustering Head**: L2-normalized features are projected to cluster logits, which
  the teacher turns into soft targets with a positionwise Sinkhorn-Knopp normalization.
- **Loss**: A cross-entropy loss between the student's predicted cluster assignments
  and the teacher's Sinkhorn-balanced assignments.

Good to Know
------------

- **Simplifications**: The example keeps the reference method faithful (inverse-block
  masking, register tokens, a cross-attention predictor with rotary position embeddings,
  and online clustering trained with its own clustering loss) but reduces the scale so it
  runs on a single GPU: fewer clusters, a shallow predictor, a small backbone, and few
  epochs. The one architectural difference is the encoder: here it is a standard masked
  ViT, whereas the reference also applies rotary position embeddings inside the encoder.
  The example also predicts all masked patches, whereas the reference subsamples a
  fraction of them each step to save compute.

Reference:
    `CAPI: Cluster and Predict Latent Patches for Improved Masked Image Modeling, 2025 <https://arxiv.org/abs/2502.08769>`_

.. note::

    CAPI requires `TIMM <https://github.com/huggingface/pytorch-image-models>`_ to be
    installed

    .. code-block:: bash

        pip install "lightly[timm]"

.. tabs::
    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/capi.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/capi.py

        .. literalinclude:: ../../../examples/pytorch/capi.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/capi.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/capi.py

        .. literalinclude:: ../../../examples/pytorch_lightning/capi.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/capi.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/capi.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Distributed Sampling is used in the dataloader
        - The Sinkhorn normalization is synchronized across processes

        Distributed Sampling makes sure that each distributed process sees only
        a subset of the data.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/capi.py
