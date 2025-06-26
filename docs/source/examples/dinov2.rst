.. _dinov2:

DINOv2
======

DINOv2 

Key Components
--------------

- **Data Augmentations**: DINO [0]_ uses random cropping, resizing, color jittering, and
  Gaussian blur to create diverse views of the same image. In particular, DINO
  generates two global views and multiple local views that are smaller crops of the
  original image.
- **Backbone**: Vision transformers, such as ViTs, and convolutional neural networks,
  such as ResNets, are employed to encode augmented images into feature representations.
- **Projection Head**: A multilayer perceptron (MLP) maps features to a distribution
  over a set of learnable clusters.
- **Distillation Loss**: The self-distillation loss encourages similarity between the
  student and teacher output distributions when processing independent views of the same
  image.

Good to Know
------------

- **Backbone Networks**: DINO [0]_ works well with transformer and convolutional neural
    network architectures.
- **Feature Quality**: DINO [0]_ learns particularly strong features without fine-tuning
    on downstream tasks. This is especially useful for clustering or
    k-Nearest Neighbors (kNN) classification.


Reference:

    .. [0] _


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
