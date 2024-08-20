.. _fastsiam:

FastSiam
========

Example implementation of the FastSiam architecture. FastSiam is an extension of the
well-known SimSiam architecture. It is a self-supervised learning method that averages
multiple target predictions to improve training with small batch sizes.

Reference:
    `FastSiam: Resource-Efficient Self-supervised Learning on a Single GPU, 2022 <https://link.springer.com/chapter/10.1007/978-3-031-16788-1_4>`_


.. tabs::
    .. tab:: PyTorch

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/fastsiam.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/fastsiam.py

        .. literalinclude:: ../../../examples/pytorch/fastsiam.py

    .. tab:: Lightning

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/fastsiam.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/fastsiam.py

        .. literalinclude:: ../../../examples/pytorch_lightning/fastsiam.py

    .. tab:: Lightning Distributed

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/fastsiam.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/fastsiam.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm

        Note that Synchronized Batch Norm is optional and the model can also be 
        trained without it. Without Synchronized Batch Norm the batch norm for 
        each GPU is only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/fastsiam.py
