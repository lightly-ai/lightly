.. _densecl:

DenseCL
=======

Example implementation of the DenseCL architecture. DenseCL is an extension of
:ref:`moco` that uses a dense contrastive loss to improve the quality of the learned
representations for object detection and segmentation tasks. While initially designed
for MoCo, DenseCL can also be combined with other self-supervised learning methods.

Reference:
    `Dense Contrastive Learning for Self-Supervised Visual Pre-Training, 2021 <https://arxiv.org/abs/2011.09157>`_


.. tabs::

    .. tab:: PyTorch

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/densecl.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/densecl.py

        .. literalinclude:: ../../../examples/pytorch/densecl.py

    .. tab:: Lightning

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/densecl.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/densecl.py

        .. literalinclude:: ../../../examples/pytorch_lightning/densecl.py

    .. tab:: Lightning Distributed

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/densecl.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/densecl.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm

        Note that Synchronized Batch Norm is optional and the model can also be 
        trained without it. Without Synchronized Batch Norm the batch norm for 
        each GPU is only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/densecl.py
