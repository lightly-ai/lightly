.. _simsiam:

SimSiam
=======

Example implementation of the SimSiam architecture.

Reference:
    `Exploring Simple Siamese Representation Learning, 2020 <https://arxiv.org/abs/2011.10566>`_

Tutorials:
    :ref:`lightly-simsiam-tutorial-4`


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/simsiam.py

        .. literalinclude:: ../../../examples/pytorch/simsiam.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/simsiam.py

        .. literalinclude:: ../../../examples/pytorch_lightning/simsiam.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/simsiam.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm

        Note that Synchronized Batch Norm is optional and the model can also be 
        trained without it.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/simsiam.py

