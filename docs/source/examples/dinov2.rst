.. _dinov2:

DINOv2
======

DINOv2 (DIstillation with NO labels v2) [0]_ is an advanced self-supervised learning framework developed by Meta AI for robust visual representation learning without labeled data. Extending the original DINO [1]_ approach, DINOv2 trains a student network to match outputs from a momentum-averaged teacher network. By leveraging self-distillation objectives at both image and patch levels, enhances both global and local feature learning. Combined with other various innovations in both the training recipe and efficient training implementation, DINOv2 exhibits state-of-the-art performance across various computer vision tasks, including classification, segmentation, and depth estimation, without the necessity for task-specific fine-tuning.

Key Components
--------------

- **Multi-level Objectives**: DINOv2 employs DINO loss for the image-level objective and iBOT [2]_ loss for patch-level objective. This multi-level approach enhances both global and local feature representations, significantly improving performance on dense prediction tasks like segmentation and depth estimation.
- **KoLeo Regularizer**: DINOv2 introduces the KoLeo regularizer [3]_, which promotes uniform spreading of features within a batch, significantly enhancing the quality of nearest-neighbor retrieval tasks without negatively affecting performance on dense downstream tasks.

Good to Know
------------

- **SOTA out-of-the-box**: DINOv2 currently represents the state-of-the-art (SOTA) among self-supervised learning (SSL) methods in computer vision, outperforming existing frameworks in various benchmarks.
- **Relation to other SSL methods**: DINOv2 can be seen as a combination of DINO and iBOT losses with the centering of SwAV [4]_.

Reference:

    .. [0] `DINOv2: Learning Robust Visual Features without Supervision, 2023 <https://arxiv.org/abs/2304.07193>`_
    .. [1] `Emerging Properties in Self-Supervised Vision Transformers, 2021 <https://arxiv.org/abs/2104.14294>`_
    .. [2] `iBOT: Image BERT Pre-Training with Online Tokenizer, 2021 <https://arxiv.org/abs/2111.07832>`_
    .. [3] `Spreading vectors for similarity search, 2018 <https://arxiv.org/abs/1806.03198>`_
    .. [4] `Unsupervised Learning of Visual Features by Contrasting Cluster Assignments, 2020 <https://arxiv.org/abs/2006.09882>`_


.. tabs::
    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/dinov2.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/dinov2.py

        .. literalinclude:: ../../../examples/pytorch/dinov2.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/dinov2.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/dinov2.py

        .. literalinclude:: ../../../examples/pytorch_lightning/dinov2.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/dinov2.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/dinov2.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm
        - Distributed Sampling is used in the dataloader

        Note that Synchronized Batch Norm is optional and the model can also be 
        trained without it. Without Synchronized Batch Norm the batch norm for 
        each GPU is only calculated based on the features on that specific GPU.
        Distributed Sampling makes sure that each distributed process sees only
        a subset of the data.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/dinov2.py
