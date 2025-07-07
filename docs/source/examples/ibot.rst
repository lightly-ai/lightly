.. _ibot:

iBOT
======

iBOT (image BERT pre-training with Online Tokenizer) [0]_ is a self-supervised learning framework for visual representation learning. It is based on masked image modeling (MIM) which draws inspiration from pretext task of masked language modeling (MLM) from NLP. It also incorporates ideas from DINO [1]_ for self-distillation and cross-view learning. More specifically, it trains a student network to reconstruct masked image patches by predicting the outputs from an online visual tokenizer (teacher). Then, iBOT jointly optimizes the tokenizer and the representation learner through a combination of masked patch prediction and cross-view self-distillation on the class token. iBOT improves the results in many downstream tasks compared to the previous methods, including image classification, object detection, instance segmentation, and semantic segmentation by showing emerging local semantic patterns.

Key Components
--------------

- **Masked Image Modeling (MIM)**:In iBOT, the student predicts feature representations of masked patches to match the teacher’s outputs, using a self-distillation objective rather than reconstructing pixel values.
- **Online Tokenizer**: iBOT introduces an online tokenizer implemented as a momentum-updated teacher network, eliminating the need for a separate offline tokenizer.
- **Cross-View Self-Distillation**: Like DINO, iBOT uses self-distillation on the [CLS] tokens from two augmented views to encourage semantic consistency.

Good to Know
------------

- **Relation to other SSL methods**: iBOT can be seen as DINO (which only uses image-level objectives) plus patch-level objectives. Further, DINOv2 [2]_ can be seen as a combination of iBOT and Koleo loss with the centering of SwAV [3]_.

Reference:

    .. [0] `iBOT: Image BERT Pre-Training with Online Tokenizer, 2021 <https://arxiv.org/abs/2111.07832>`_
    .. [1] `Emerging Properties in Self-Supervised Vision Transformers, 2021 <https://arxiv.org/abs/2104.14294>`_
    .. [2] `DINOv2: Learning Robust Visual Features without Supervision, 2023 <https://arxiv.org/abs/2304.07193>`_
    .. [3] `Unsupervised Learning of Visual Features by Contrasting Cluster Assignments, 2020 <https://arxiv.org/abs/2006.09882>`_

.. tabs::
    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/ibot.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/ibot.py

        .. literalinclude:: ../../../examples/pytorch/ibot.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/ibot.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/ibot.py

        .. literalinclude:: ../../../examples/pytorch_lightning/ibot.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/ibot.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/ibot.py

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

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/ibot.py
