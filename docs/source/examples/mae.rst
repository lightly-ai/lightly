.. _mae:

MAE
===

Example implementation of the Masked Autoencoder (MAE) method. MAE is a
transformer-based method that leverages a `Vision Transformer (ViT)
<https://arxiv.org/abs/2010.11929>`_ as its backbone to learn image
representations by predicting pixel values of masked patches. As an autoencoder,
MAE consists of an encoder that processes masked images to generate latent
representations and a decoder that reconstructs the input images from these
representations. The masking operation significantly reduces the sequence length
processed by the transformer encoder, which improves computational efficiency
compared to other transformer-based self-supervised learning methods. By
reconstructing the masked patches, MAE effectively forces the model to learn
meaningful representations of the data.



Key Components
--------------

- **Data Augmentations**:  Unlike contrastive and most self-distillation methods, MAE minimizes reliance on handcrafted data augmentations. The only augmentation used is random resized cropping.
- **Masking**: MAE applies masking to 75% of the input patches, meaning only 25% of the image tokens are fed into the transformer encoder. 
- **Backbone**: MAE employs a standard ViT to encode the masked images.
- **Decoder**: The decoder processes visible tokens alongside shared, learnable mask tokens. It reconstructs the original input image by predicting the pixel values of the masked patches.
- **Reconstruction Loss**: A Mean Squared Error (MSE) loss is applied between the original and reconstructed pixel values of the masked patches.

Good to Know
----------------

- **Backbone Networks**: The masking process used by MAE is inherently incompatible with convolutional-based architectures.
- **Computational Efficiency**: The masking mechanism allows the encoder to process only a subset of the image tokens, significantly reducing computational overhead.
- **Scalability**: MAE demonstrates excellent scalability with respect to both model and data size as demonstrated `here. <https://arxiv.org/abs/2303.13496>`_
- **Versatility**: The minimal reliance on handcrafted data augmentations makes MAE adaptable to diverse data domains. For example, its application in medical imaging is discussed in `this study. <https://arxiv.org/abs/2203.05573>`_
- **Shallow Evaluations**: Despite their strong performance in the fine-tuning regime, models trained with MAE tend to underperform in shallow evaluations, such as k-NN or linear evaluation with a frozen backbone.

Reference:
    `Masked Autoencoders Are Scalable Vision Learners, 2021 <https://arxiv.org/abs/2111.06377>`_

.. note::

    MAE requires `TIMM <https://github.com/huggingface/pytorch-image-models>`_ to be
    installed

    .. code-block:: bash

        pip install "lightly[timm]"

.. tabs::
    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/mae.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/mae.py

        .. literalinclude:: ../../../examples/pytorch/mae.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/mae.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/mae.py

        .. literalinclude:: ../../../examples/pytorch_lightning/mae.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/mae.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/mae.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Distributed Sampling is used in the dataloader

        Distributed Sampling makes sure that each distributed process sees only
        a subset of the data.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/mae.py
