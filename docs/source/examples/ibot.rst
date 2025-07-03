.. _ibot:

iBOT
======

iBOT (image BERT pre-training with Online Tokenizer) [0]_ is a self-supervised learning framework for visual representation learning based on masked image modeling (MIM) and self-distillation. It trains a student network to reconstruct masked image patches by predicting the outputs from an online tokenizer, implemented as a momentum-updated teacher network, thereby eliminating the need for offline pre-trained tokenizers. iBOT jointly optimizes the tokenizer and the representation learner through a combination of masked patch prediction and cross-view self-distillation on the class token. Key components include progressive learning of semantically meaningful visual tokens, block-wise masking augmentation, and a shared projection head for improved feature abstraction. iBOT achieves state-of-the-art results in image classification, robustness to image corruptions, and dense prediction tasks, highlighting its ability to capture rich local semantics and robust visual representations.

Key Components
--------------


- **Online Tokenizer**: iBOT introduces an online tokenizer implemented as a momentum-updated teacher network, eliminating the need for a separate offline tokenizer.
- **Masked Image Modeling (MIM)**: iBOT performs masked prediction of image patches using a self-distillation objective, where the student network reconstructs masked tokens based on predictions from the teacher network.
- **Cross-View Self-Distillation**: Similar to DINO [1]_, iBOT applies self-distillation to the [CLS] tokens of two augmented views of the same image, promoting semantic abstraction.
- **Projection Head**: A shared multilayer perceptron (MLP) projects both patch and [CLS] tokens into a high-dimensional embedding space, enabling effective knowledge transfer and feature representation.
- **Block-Wise Masking**: iBOT employs random block-wise masking to generate diverse training samples, facilitating richer context learning.


Good to Know
------------

- **Tokenization Strategy**: iBOT does not require pre-trained offline tokenizers, as tokenization and representation learning are conducted jointly online.
- **Semantic Representation**: iBOT learns semantically rich visual representations, improving robustness against common corruptions and benefiting downstream dense tasks like object detection and semantic segmentation.


Reference:

    .. [0] `iBOT: Image BERT Pre-Training with Online Tokenizer, 2021 <https://arxiv.org/abs/2111.07832>`_
    .. [1] `Emerging Properties in Self-Supervised Vision Transformers, 2021 <https://arxiv.org/abs/2104.14294>`_

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
