.. _simclr:

SimCLR
======

Example implementation of the SimCLR architecture.

Reference:
    `A Simple Framework for Contrastive Learning of Visual Representations, 2020 <https://arxiv.org/abs/2002.05709>`_

Tutorials:
    :ref:`lightly-simclr-tutorial-3`


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/simclr.py

        .. literalinclude:: ../../../examples/pytorch/simclr.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/simclr.py

        .. literalinclude:: ../../../examples/pytorch_lightning/simclr.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/simclr.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm
        - Features are gathered from all GPUs before the loss is calculated

        Note that Synchronized Batch Norm and feature gathering are optional and
        the model can also be trained without them.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/simclr.py
