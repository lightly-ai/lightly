.. _nnclr:

NNCLR
=====

Example implementation of the NNCLR architecture.

Reference:
    `With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations, 2021 <https://arxiv.org/abs/2104.14548>`_

.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/nnclr.py

        .. literalinclude:: ../../../examples/pytorch/nnclr.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/nnclr.py

        .. literalinclude:: ../../../examples/pytorch_lightning/nnclr.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/nnclr.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm

        Note that Synchronized Batch Norm is optional and the model can also be 
        trained without it. Without Synchronized Batch Norm the batch norm for 
        each GPU is only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/nnclr.py

