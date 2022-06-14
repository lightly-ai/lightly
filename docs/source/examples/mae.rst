.. _mae:

MAE
===

Example implementation of the Masked Autoencoder (MAE) architecture. MAE is a
transformer model based on the `Vision Transformer (ViT) <https://arxiv.org/abs/2010.11929>`_ 
architecture. It learns image representations by predicting pixel values for
masked patches on the input images. The network is split into an encoder and
decoder. The encoder generates the image representation and the decoder predicts
the pixel values from the representation. MAE outperforms other self-supervised
learning approaches on several classification, object detection and semantic
segmentation benchmarks. It also increases training efficiency compared to
other transformer architectures by encoding only part of the input image and
using a shallow decoder architecture.

Reference:
    `Masked Autoencoders Are Scalable Vision Learners, 2021 <https://arxiv.org/abs/2111.06377>`_


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/mae.py

        .. literalinclude:: ../../../examples/pytorch/mae.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/mae.py

        .. literalinclude:: ../../../examples/pytorch_lightning/mae.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/mae.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Distributed Sampling is used in the dataloader

        Distributed Sampling makes sure that each distributed process sees only
        a subset of the data.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/mae.py
