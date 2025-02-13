.. _simsiam:

SimSiam
=======

SimSiam (Simple Siamese Representation Learning) [0]_ is a self-supervised learning framework for visual representation learning that eliminates the need for negative samples, large batches, or momentum encoders. Instead, SimSiam directly optimizes the similarity between two augmented views of an image. It employs a simple Siamese architecture where these augmented views are processed by a shared encoder, with a prediction MLP on one branch and a stop-gradient operation on the other. SimSiam challenges conventional beliefs regarding collapsing representations by demonstrating that only the stop-gradient mechanism is essential for preventing collapse, rather than relying on momentum encoders or architectural modifications. Experimental results highlight the crucial role of the predictor layer and the application of batch normalization in hidden layers for stable training and improved representation quality. Furthermore, unlike SimCLR [1]_ and SwAV [2]_, SimSiam performs robustly across a wide range of batch sizes.

Key Components
--------------

- **Data Augmentations**: SimSiam employs the same augmentations as SimCLR, including random resized cropping, horizontal flipping, color jittering, Gaussian blur, and solarization. These augmentations provide diverse views of an image for representation learning.
- **Backbone**: SimSiam utilizes ResNet-type architectures as the encoder network. The model does not employ a momentum encoder.
- **Projection & Prediction Head**: A projection MLP maps the encoder output to a lower-dimensional space, followed by a prediction MLP on one branch. The stop-gradient operation is applied to the second branch to prevent collapse.
- **Loss Function**: SimSiam minimizes the negative cosine similarity between the predicted representation of one view and the projected representation of the other, with a symmetrical loss formulation. It also works for a symmetrized cross-entropy loss.

Good to Know
-------------

- **Backbone Networks**: SimSiam is specifically optimized for convolutional neural networks, with a focus on ResNet architectures. We do not recommend using it with transformer-based models and instead suggest using DINO [3]_.
- **Relation to SimCLR**: SimSiam can be thought of as "SimCLR without negative pairs."
- **Relation to SwAV**: SimSiam is conceptually analogous to “SwAV without online clustering.”
- **Relation to BYOL** [4]_: SimSiam can be considered a variation of BYOL that removes the momentum encoder subject to many implementation differences.

Reference:
    .. [0] `Exploring Simple Siamese Representation Learning, 2020 <https://arxiv.org/abs/2011.10566>`_
    .. [1] `A Simple Framework for Contrastive Learning of Visual Representations, 2020 <https://arxiv.org/abs/2002.05709>`_
    .. [2] `Unsupervised Learning of Visual Features by Contrasting Cluster Assignments, 2020 <https://arxiv.org/abs/2006.09882>`_
    .. [3] `Emerging Properties in Self-Supervised Vision Transformers, 2021 <https://arxiv.org/abs/2104.14294>`_
    .. [4] `Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning, 2020 <https://arxiv.org/abs/2006.07733>`_


Tutorials:
    :ref:`lightly-simsiam-tutorial-4`


.. tabs::
    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/simsiam.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/simsiam.py

        .. literalinclude:: ../../../examples/pytorch/simsiam.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/simsiam.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/simsiam.py

        .. literalinclude:: ../../../examples/pytorch_lightning/simsiam.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/simsiam.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/simsiam.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm

        Note that Synchronized Batch Norm is optional and the model can also be 
        trained without it. Without Synchronized Batch Norm the batch norm for 
        each GPU is only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/simsiam.py

