.. _beit:

BEIT
====

implementation of the BEIT (BERT Pre-Training of Image Transformers) 
architecture for masked image modeling (MIM).

BEIT pre-trains a Vision Transformer by masking random patches of the input 
image and predicting the discrete visual tokens of the masked patches. Unlike 
MAE which predicts raw pixels, BEIT uses a pre-trained discrete VAE tokenizer 
to convert images into a vocabulary of visual tokens, and the transformer learns 
to predict these token indices.

Key components:

- **Blockwise masking**: Random block-shaped regions are masked rather than 
  individual patches, following the BEIT paper.
- **Discrete VAE tokenizer**: A pre-trained tokenizer (e.g., DALL-E dVAE) 
  converts images into visual token targets.
- **Learnable mask token**: A special [M] embedding replaces masked patches 
  in the input.
- **Shared relative position bias**: Optional relative position bias can be 
  used instead of absolute position embeddings.

Reference:
    `BEIT: BERT Pre-Training of Image Transformers, 2021 <https://arxiv.org/abs/2106.08254>`_


.. tabs::
    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/beit.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/beit.py

        .. literalinclude:: ../../../examples/pytorch/beit.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/beit.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/beit.py

        .. literalinclude:: ../../../examples/pytorch_lightning/beit.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/beit.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/beit.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm

        Note that Synchronized Batch Norm is optional and the model can also be 
        trained without it. Without Synchronized Batch Norm the batch norm for 
        each GPU is only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/beit.py