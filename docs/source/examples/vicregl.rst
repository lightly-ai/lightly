.. _vicreg:

VICRegL
=======

VICRegL (VICRegL: Self-Supervised Learning of Local Visual Features) is a method derived from `VICReg, 2022 <https://arxiv.org/abs/2105.04906>`_.
As the standard VICReg, it avoids the collapse problem with a simple regularization term on the variance of the embeddings along each dimension individually. 
Moreover, it learns good global and local features simultaneously, yielding excellent performance on detection and segmentation tasks while maintaining good performance on classification tasks. 

Reference:
    `VICRegL: Self-Supervised Learning of Local Visual Features, 2022 <https://arxiv.org/abs/2210.01571>`_


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/vicregl.py

        .. literalinclude:: ../../../examples/pytorch/vicregl.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/vicregl.py

        .. literalinclude:: ../../../examples/pytorch_lightning/vicregl.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/vicregl.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Distributed Sampling is used in the dataloader

        Distributed Sampling makes sure that each distributed process sees only
        a subset of the data.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/vicregl.py