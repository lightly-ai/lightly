.. _barlowtwins:


Barlow Twins
============

Example implementation of the Barlow Twins architecture.

Reference:
    `Barlow Twins: Self-Supervised Learning via Redundancy Reduction, 2021 <https://arxiv.org/abs/2103.03230>`_


.. tabs::

    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/barlowtwins.py

        .. literalinclude:: ../../../examples/pytorch/barlowtwins.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/barlowtwins.py

        .. literalinclude:: ../../../examples/pytorch_lightning/barlowtwins.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/barlowtwins.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm
        - Features are gathered from all GPUs before the loss is calculated

        Note that Synchronized Batch Norm and feature gathering are optional and
        the model can also be trained without them.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/barlowtwins.py
