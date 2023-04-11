.. _fastsiam:

FastSiam
=======

Example implementation of the FastSiam architecture.

Reference:
    `FastSiam: Resource-Efficient Self-supervised Learning on a Single GPU, 2022 <https://link.springer.com/chapter/10.1007/978-3-031-16788-1_4>`_


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/fastsiam.py

        .. literalinclude:: ../../../examples/pytorch/fastsiam.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/fastsiam.py

        .. literalinclude:: ../../../examples/pytorch_lightning/fastsiam.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/fastsiam.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm

        Note that Synchronized Batch Norm is optional and the model can also be 
        trained without it. Without Synchronized Batch Norm the batch norm for 
        each GPU is only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/fastsiam.py
