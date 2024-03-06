.. _aim:

AIM
===

Example implementation of the Autoregressive Image Model (AIM) architecture. AIM is a
transformer model based on the `Vision Transformer (ViT) <https://arxiv.org/abs/2010.11929>`_
architecture. It learns image representations by predicting pixel values for image
patches based on previous patches in the image. This is similar to the next word prediction
task in natural language processing. AIM demonstrates that it is possible to train
large-scale vision models using an autoregressive objective. The model is split into
and encoder and a decoder part. The encoder generates features for image patches and
the decoder predicts pixel values based on the features.

Reference:
    `Scalable Pre-training of Large Autoregressive Image Models, 2024 <https://arxiv.org/abs/2401.08541>`_


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/aim.py

        .. literalinclude:: ../../../examples/pytorch/aim.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/aim.py

        .. literalinclude:: ../../../examples/pytorch_lightning/aim.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/aim.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Distributed Sampling is used in the dataloader

        Distributed Sampling makes sure that each distributed process sees only
        a subset of the data.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/aim.py
