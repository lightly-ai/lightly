.. _detcon:

DetConS
=======

Example implementation of the DetConS architecture. DetConS is an extension of
:ref:`simclr` that contrasts per-region embeddings instead of whole-image embeddings,
using unsupervised segmentation masks to define regions. This allows the model to
learn representations that are better suited for object detection and segmentation tasks.

Reference:
    `Efficient Visual Pretraining with Contrastive Detection, 2021 <https://arxiv.org/abs/2103.10957>`_


.. tabs::

    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/detcon.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/detcon.py

        .. literalinclude:: ../../../examples/pytorch/detcon.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/detcon.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/detcon.py

        .. literalinclude:: ../../../examples/pytorch_lightning/detcon.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/detcon.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/detcon.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm

        Note that Synchronized Batch Norm is optional and the model can also be
        trained without it. Without Synchronized Batch Norm the batch norm for
        each GPU is only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/detcon.py
