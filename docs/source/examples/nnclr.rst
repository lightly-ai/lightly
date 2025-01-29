.. _nnclr:

NNCLR
=====

NNCLR is a self-supervised framework for visual representation learning using contrastive methods.
It is similar to SimCLR by using two augmented views of the same image, prediction and projection heads and a contrastive loss.
It differs from SimCLR by replacing each sample with its nearest neightbour in a support set or memory bank.
Furthermore, it makes the loss symmetric and uses other sizes of the projection and prediction head layers.

NNCLR shows a significant improvement on all use cases of self-supervised learning:
Compared to both SimCLR and other self-supervised learning methods,
it has a better ImageNet linear evaluation performance, and also shows better semi-supervised learning results. 
On transfer learning tasks, NNCLR is the best performing method compared to all of BYOL, SimCLR and supervised pretraining on ImageNet in 8 out of 12 cases.


Key Components
--------------

- **Data Augmentations**: Exactly like SimCLR, NNCLR uses random cropping, resizing, color jittering, and Gaussian blur to create diverse views of the same image.
- **Backbone**: Convolutional neural networks, such as ResNet, are employed to encode augmented images into feature representations.
- **Projection Head**: A multilayer perceptron (MLP) maps features into a space where contrastive loss is applied, enhancing representation quality.
- **Memory Bank**: NNCLR uses a first-in first-out memory bank to store the features of previous samples. The features of each batch are stored in the memory bank and the oldest features are then discarded. The size of the memory bank is kept large enough to approximate the full dataset.
- **Nearest Neighbour Sampling**: The features of each view are replaced by the features of their nearest neighbour in the memory bank. This provides more semantic variations than pre-defined data augmentations.
- **Contrastive Loss**: The normalized temperature-scaled cross-entropy loss (NT-Xent) encourages similar pairs to align and dissimilar pairs to diverge.

Good to Know
----------------

- **Backbone Networks**: NNCLR is specifically optimized for convolutional neural networks, with a focus on ResNet architectures. We do not recommend using it with transformer-based models.
- **Learning Paradigm**: NNCLR is less dependent on good augmentations than SimCLR, as the nearest neighbour sampling provides more semantic variations. However, the method still benefits from a good choice of augmentations and larger batch sizes.

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

