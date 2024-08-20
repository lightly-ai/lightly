.. _byol:

BYOL
====

Example implementation of the BYOL architecture.

Reference:
    `Bootstrap your own latent: A new approach to self-supervised Learning, 2020 <https://arxiv.org/abs/2006.07733>`_


.. tabs::

    .. tab:: PyTorch

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/byol.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/byol.py

        .. literalinclude:: ../../../examples/pytorch/byol.py

    .. tab:: Lightning

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/byol.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/byol.py

        .. literalinclude:: ../../../examples/pytorch_lightning/byol.py

    .. tab:: Lightning Distributed

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/byol.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/byol.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm

        Note that Synchronized Batch Norm is optional and the model can also be 
        trained without it. Without Synchronized Batch Norm the batch norm for 
        each GPU is only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/byol.py
