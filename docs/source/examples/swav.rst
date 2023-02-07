.. _swav:

SwaV
====

Example implementation of the SwaV architecture. This model takes advantage of contrastive methods without requiring to compute pairwise comparisons. 
Specifically, this method simultaneously clusters the data while enforcing consistency between cluster assignments produced for different augmentations of the same image,
instead of comparing features directly as in contrastive learning. It can be trained with large and small batch sizes.

Reference:
    `Unsupervised Learning of Visual Features by Contrasting Cluster Assignments, 2020 <https://arxiv.org/abs/2006.09882>`_


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/swav.py

        .. literalinclude:: ../../../examples/pytorch/swav.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/swav.py

        .. literalinclude:: ../../../examples/pytorch_lightning/swav.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/swav.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm
        - Distributed Sinkhorn is used in the loss calculation 

        Note that Synchronized Batch Norm and distributed Sinkhorn are optional 
        and the model can also be trained without them. Without Synchronized 
        Batch Norm and distributed Sinkhorn the batch norm and loss for each GPU 
        are only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/swav.py

If you are planning to work with small batch sizes (less than 256), please use the SwaV implementation with queue:

.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/swav_queue.py

        .. literalinclude:: ../../../examples/pytorch/swav_queue.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/swav_queue.py

        .. literalinclude:: ../../../examples/pytorch_lightning/swav_queue.py
