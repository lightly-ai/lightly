.. _tico:

TiCo
====

Example implementation of Transformation Invariance and Covariance Contrast (TiCo)
for self-supervised visual representation learning. Similar to BYOL, this method is based on maximizing 
the agreement among embeddings of different distorted versions of the same image, which pushes the encoder to 
produce transformation invariant representations.

Reference:
    `TiCo: Transformation Invariance and Covariance Contrast for Self-Supervised Visual Representation Learning, 2022 <https://arxiv.org/pdf/2206.10698.pdf>`_


.. tabs::

    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/tico.py

        .. literalinclude:: ../../../examples/pytorch/tico.py

    .. tab:: Lighning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/tico.py

        .. literalinclude:: ../../../examples/pytorch_lightning/tico.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/tico.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm

        Note that Synchronized Batch Norm is optional and the model can also be 
        trained without it. Without Synchronized Batch Norm the batch norm for 
        each GPU is only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/tico.py
