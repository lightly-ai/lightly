.. _barlowtwins:


Barlow Twins
============

Example implementation of the Barlow Twins architecture.

Reference:
    `Barlow Twins: Self-Supervised Learning via Redundancy Reduction, 2021 <https://arxiv.org/abs/2103.03230>`_


.. tabs::

    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/barlowtwins.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/barlowtwins.py

        .. literalinclude:: ../../../examples/pytorch/barlowtwins.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/barlowtwins.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/barlowtwins.py

        .. literalinclude:: ../../../examples/pytorch_lightning/barlowtwins.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/barlowtwins.ipynb

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
        the model can also be trained without them. Without Synchronized Batch
        Norm and feature gathering the batch norm and loss for each GPU are 
        only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/barlowtwins.py
