.. _simmim:

SimMIM
======

Example implementation of SimMIM: A Simple Framework for Masked Image Modeling architecture. SimMIM is a
very similar architecture to `Masked Autoencoders Are Scalable Vision Learners, 2021 <https://arxiv.org/abs/2111.06377>`_.
It uses a ViT encoder using as input both masked and non-masked patches. Other differences with respect to MAE
is that it has just a simple linear layer as a decoder and uses L1 instead of L2 loss.

Reference:
    `SimMIM: A Simple Framework for Masked Image Modeling, 2021 <https://arxiv.org/abs/2111.09886>`_


.. tabs::
    .. tab:: PyTorch

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/simmim.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/simmim.py

        .. literalinclude:: ../../../examples/pytorch/simmim.py

    .. tab:: Lightning

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/simmim.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/simmim.py

        .. literalinclude:: ../../../examples/pytorch_lightning/simmim.py

    .. tab:: Lightning Distributed

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/simmim.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/simmim.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Distributed Sampling is used in the dataloader

        Distributed Sampling makes sure that each distributed process sees only
        a subset of the data.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/simmim.py
