.. _simdino:

SimDINO
=======

SimDINO [0]_ simplifies DINO [1]_ by replacing most of its heuristics with an explicit
coding rate term. Like DINO, a student network learns to match a momentum-averaged
teacher on different views of the same image. Instead of a prototype layer, centering,
and temperature sharpening, SimDINO works directly on the l2-normalized projection head
outputs: a cosine similarity term pulls the student and teacher views together, and a
coding rate term on the global views spreads the features out to prevent collapse.
The authors report that this makes training more robust to design choices such as the
architecture and hyperparameters, and that it learns higher-quality representations
on downstream tasks.


Key Components
--------------

- **Data Augmentations**: SimDINO uses the same augmentations as DINO: two global
  views and multiple smaller local views of each image.
- **Backbone**: Vision transformers and convolutional neural networks can be used to
  encode the augmented images.
- **Projection Head**: The DINO projection head without the weight-normalized prototype
  layer. Its outputs are l2-normalized.
- **Loss**: The student and teacher outputs are aligned with a cosine similarity term,
  and the coding rate of the global view features is maximized.

Good to Know
------------

- **Fewer hyperparameters**: SimDINO has no teacher or student temperatures, no
  centering, and no prototype layer to freeze during the first epochs.
- **Relation to other SSL methods**: The coding rate term is the total coding rate also
  used by EMP-SSL [2]_. SimDINOv2 applies the same simplification to DINOv2.


Reference:

    .. [0] `Simplifying DINO via Coding Rate Regularization, 2025 <https://arxiv.org/abs/2502.10385>`_
    .. [1] `Emerging Properties in Self-Supervised Vision Transformers, 2021 <https://arxiv.org/abs/2104.14294>`_
    .. [2] `EMP-SSL: Towards Self-Supervised Learning in One Training Epoch, 2023 <https://arxiv.org/abs/2304.03977>`_


.. tabs::
    .. tab:: PyTorch

        .. image:: /_static/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/simdino.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/simdino.py

        .. literalinclude:: ../../../examples/pytorch/simdino.py

    .. tab:: Lightning

        .. image:: /_static/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/simdino.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/simdino.py

        .. literalinclude:: ../../../examples/pytorch_lightning/simdino.py

    .. tab:: Lightning Distributed

        .. image:: /_static/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/simdino.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/simdino.py

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

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/simdino.py
