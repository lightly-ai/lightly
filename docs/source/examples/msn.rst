.. _msn:

MSN
===

Example implementation of the Masked Siamese Networks (MSN) architecture. MSN is a
transformer model based on the `Vision Transformer (ViT) <https://arxiv.org/abs/2010.11929>`_ 
architecture. It learns image representations by comparing cluster assignments of
masked and unmasked image views. The network is split into a target and anchor network.
The target network creates representations from unmasked image views while the anchor
network receives a masked image view. MSN increases training efficiency as the backward
pass is only calculated for the anchor network. The target network is updated via
momentum from the anchor network.

Reference:
    `Masked Siamese Networks for Label-Efficient Learning, 2022 <https://arxiv.org/abs/2204.07141>`_


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/msn.py

        .. literalinclude:: ../../../examples/pytorch/msn.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/msn.py

        .. literalinclude:: ../../../examples/pytorch_lightning/msn.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/msn.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Distributed Sampling is used in the dataloader
        - Distributed Sinkhorn is used in the loss calculation 

        Distributed Sampling makes sure that each distributed process sees only
        a subset of the data.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/msn.py
