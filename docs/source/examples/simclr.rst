.. _simclr:

SimCLR
======

SimCLR is a framework for self-supervised learning of visual representations using contrastive learning. It aims to maximize agreement between different augmented views of the same image.

Key Components
--------------

- **Data Augmentations**: SimCLR uses random cropping, resizing, color jittering, and Gaussian blur to create diverse views of the same image.
- **Backbone**: Convolutional neural networks, such as ResNet, are employed to encode augmented images into feature representations.
- **Projection Head**: A multilayer perceptron (MLP) maps features into a space where contrastive loss is applied, enhancing representation quality.
- **Contrastive Loss**: The normalized temperature-scaled cross-entropy loss (NT-Xent) encourages similar pairs to align and dissimilar pairs to diverge.

Good to Know
----------------

- **Backbone Networks**: SimCLR is specifically optimized for convolutional neural networks, with a focus on ResNet architectures. We do not recommend using it with transformer-based models.
- **Learning Paradigm**: SimCLR is based on contrastive learning which makes it sensitive to the augmentations you pick and the method benefits from larger batch sizes.

Reference:
    `A Simple Framework for Contrastive Learning of Visual Representations, 2020 <https://arxiv.org/abs/2002.05709>`_

Tutorials:
    :ref:`lightly-simclr-tutorial-3`


.. tabs::
    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/simclr.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/simclr.py

        .. literalinclude:: ../../../examples/pytorch/simclr.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/simclr.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/simclr.py

        .. literalinclude:: ../../../examples/pytorch_lightning/simclr.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/simclr.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/simclr.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm
        - Features are gathered from all GPUs before the loss is calculated

        Note that Synchronized Batch Norm and feature gathering are optional and
        the model can also be trained without them. Without Synchronized Batch
        Norm and feature gathering the batch norm and loss for each GPU are 
        only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/simclr.py
