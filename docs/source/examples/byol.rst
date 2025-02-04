.. _byol:

BYOL
====

BYOL (Bootstrap Your Own Latent) [0]_ is a self-supervised learning framework for visual 
representation learning without negative samples. Unlike contrastive learning methods, 
such as MoCo [1]_ and SimCLR [2]_ that compare positive and negative pairs, BYOL uses 
two neural networks – "online" and a "target" networks – where the online network is 
trained to predict the target’s representations of the same image under different 
augmentations. The target's weights are updated as the exponential moving average 
(EMA) of the online network, and the authors show that this is enough to prevent 
collapse to trivial solutions. The authors particularly show that due to the absence
of negative samples, BYOL is less sensitive to the batch size during training and manages
to achieve state-of-the-art on several semi-supervised and transfer learning benchmarks.

Key Components
--------------

- **Data Augmentations**: BYOL [0]_ uses the same augmentations as SimCLR [2]_, namely
    random resized crop, random horizontal flip, color distortions, Gaussian blur and
    solarization. The color distortiion consits of a random sequence of brightness,
    constrast, saturation, hue adjustments and an optional grayscale conversion. However
    the hyperparameters for the augmentations are different from SimCLR [2]_.
- **Backbone**: BYOL [0]_ uses ResNet-type convolutional backbones as the online and
    target networks. They do not evaluate the performance of other architectures.
- **Projection & Prediction Head**: A projection head is used to map the output of the
    backbone to a lower-dimensional space. The target network once again relies on an
    EMA of the online network's projection head for the projection head. A notable
    architectureal choice is the use of an additional prediction head, a secondary MLP
    appended to only the online network's projection head.
- **Loss Function**: BYOL [0]_ uses a negative cosine similarity loss between the
    normalized representations of the online's prediction output and the targe's
    projection output.


Reference:
    .. [0] `Bootstrap your own latent: A new approach to self-supervised Learning, 2020 <https://arxiv.org/abs/2006.07733>`_
    .. [1] `Momentum Contrast for Unsupervised Visual Representation Learning, 2019 <https://arxiv.org/abs/1911.05722>`_
    .. [2] `A Simple Framework for Contrastive Learning of Visual Representations, 2020 <https://arxiv.org/abs/2002.05709>`_


.. tabs::

    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/byol.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/byol.py

        .. literalinclude:: ../../../examples/pytorch/byol.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/byol.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/byol.py

        .. literalinclude:: ../../../examples/pytorch_lightning/byol.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/byol.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/byol.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm

        Note that Synchronized Batch Norm is optional and the model can also be 
        trained without it. Without Synchronized Batch Norm the batch norm for 
        each GPU is only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/byol.py
