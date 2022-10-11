.. _smog:

SMoG
====

Example implementation of the Synchronous Momentum Grouping (SMoG) paper. 
SMoG follows the framework of contrastive learning but replaces the contrastive
unit from instance to group, mimicking clustering-based methods. To
achieve this, they propose the momentum grouping scheme which synchronously 
conducts feature grouping with representation learning. 

Reference:
    `Unsupervised Visual Representation Learning by Synchronous Momentum Grouping, 2022 <https://arxiv.org/pdf/2207.06167.pdf>`_


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/smog.py

        .. literalinclude:: ../../../examples/pytorch/smog.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/smog.py

        .. literalinclude:: ../../../examples/pytorch_lightning/smog.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/smog.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Distributed Sampling is used in the dataloader
        - Distributed Sinkhorn is used in the loss calculation 

        Distributed Sampling makes sure that each distributed process sees only
        a subset of the data.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/smog.py
