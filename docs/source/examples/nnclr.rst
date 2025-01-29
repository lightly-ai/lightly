.. _nnclr:

NNCLR
=====

NNCLR is a self-supervised framework for visual representation learning that builds upon contrastive methods. It shares similarities with SimCLR, such as using two augmented views of the same image, projection and prediction heads, and a contrastive loss. However, it introduces key modifications:

1. Nearest Neighbor Replacement: Instead of directly comparing two augmented views of the same sample, NNCLR replaces each sample with its nearest neighbor in a support set (or memory bank). This increases semantic variation in the learned representations.
2. Symmetric Loss: The contrastive loss is made symmetric to improve training stability.
3. Architectural Adjustments: NNCLR employs different sizes for projection and prediction head layers compared to SimCLR.

These improvements result in significantly better performance across multiple self-supervised learning benchmarks. Compared to SimCLR and other self-supervised methods, NNCLR achieves:
- Higher ImageNet linear evaluation accuracy.
- Improved semi-supervised learning results.
- Superior performance on transfer learning tasks, outperforming BYOL, SimCLR, and even supervised ImageNet pretraining in 8 out of 12 benchmarked cases.

Key Components
--------------

- **Data Augmentations**: NNCLR applies the same transformations as SimCLR, including random cropping, resizing, color jittering, and Gaussian blur, to create diverse views of the same image.
- **Backbone**: A convolutional neural network (typically ResNet) encodes augmented images into feature representations.
- **Projection Head**: A multilayer perceptron (MLP) maps features into a contrastive space, improving representation learning.
- **Memory Bank**: NNCLR maintains a first-in, first-out (FIFO) memory bank, storing past feature representations. Older features are gradually discarded, ensuring a large and diverse set approximating the full dataset.
- **Nearest Neighbor Sampling**: Each feature representation is replaced by its nearest neighbor from the memory bank, introducing additional semantic variation beyond standard augmentations.
- **Contrastive Loss**: NNCLR employs normalized temperature-scaled cross-entropy loss (NT-Xent), encouraging alignment between positive pairs and separation from negative pairs.

Good to Know
----------------

- **Optimized for CNNs**: NNCLR is specifically designed for convolutional neural networks (CNNs), particularly ResNet. It is not recommended for transformer-based architectures.
- **Augmentation Robustness**: Compared to SimCLR, NNCLR is less dependent on strong augmentations since nearest neighbor sampling introduces natural semantic variation. However, performance still benefits from well-chosen augmentations and larger batch sizes.


Reference:
    `With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations, 2021 <https://arxiv.org/abs/2104.14548>`_

.. tabs::
    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/nnclr.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/nnclr.py

        .. literalinclude:: ../../../examples/pytorch/nnclr.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/nnclr.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/nnclr.py

        .. literalinclude:: ../../../examples/pytorch_lightning/nnclr.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/nnclr.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/nnclr.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm

        Note that Synchronized Batch Norm is optional and the model can also be 
        trained without it. Without Synchronized Batch Norm the batch norm for 
        each GPU is only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/nnclr.py

