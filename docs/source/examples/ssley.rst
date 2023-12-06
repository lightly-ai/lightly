.. _ssley:

SSL-EY
=======

SSL-EY is a method that explicitly
avoids the collapse problem with a simple regularization term on the variance of the embeddings along each dimension individually. It inherits the model structure from
`Barlow Twins, 2022 <https://arxiv.org/abs/2103.03230>`_ changing the loss. Doing so allows the stabilization of the training and leads to performance improvements. 

Reference:
    `Efficient Algorithms for the CCA Family: Unconstrained Objectives with Unbiased Gradients, 2023 <https://arxiv.org/abs/2310.01012>`_


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/ssley.py

        .. literalinclude:: ../../../examples/pytorch/ssley.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/ssley.py

        .. literalinclude:: ../../../examples/pytorch_lightning/ssley.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/ssley.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Distributed Sampling is used in the dataloader

        Distributed Sampling makes sure that each distributed process sees only
        a subset of the data.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/ssley.py