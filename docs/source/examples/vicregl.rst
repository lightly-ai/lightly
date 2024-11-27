.. _vicregl:

VICRegL
=======

VICRegL (VICRegL: Self-Supervised Learning of Local Visual Features) is a method derived from `VICReg, 2022 <https://arxiv.org/abs/2105.04906>`_.
As the standard VICReg, it avoids the collapse problem with a simple regularization term on the variance of the embeddings along each dimension individually. 
Moreover, it learns good global and local features simultaneously, yielding excellent performance on detection and segmentation tasks while maintaining good performance on classification tasks. 

Reference:
    `VICRegL: Self-Supervised Learning of Local Visual Features, 2022 <https://arxiv.org/abs/2210.01571>`_


.. tabs::
    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/vicregl.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/vicregl.py

        .. literalinclude:: ../../../examples/pytorch/vicregl.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/vicregl.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/vicregl.py

        .. literalinclude:: ../../../examples/pytorch_lightning/vicregl.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/vicregl.ipynb

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