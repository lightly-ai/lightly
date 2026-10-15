.. _franca:

Franca
======

Franca [0]_ is a self-supervised learning method in the DINOv2 [1]_ lineage. Like DINOv2, it trains a student network to match a momentum-averaged teacher with a DINO [2]_ image-level objective and an iBOT [3]_ patch-level objective. Franca adds nested Matryoshka [4]_ clustering: the clustering objectives run on several nested prefixes of the embedding at once, so shorter prefixes stay usable on their own. It also masks the iBOT branch with a cyclic block mask.

Key Components
--------------

- **Matryoshka nested clustering**: The projection head runs a separate DINOv2-style clustering head on each nested prefix of the backbone embedding, with a prototype count that scales with the nested dimension. The DINO and iBOT losses are computed per nesting level and summed, so a single backbone learns representations that stay useful when truncated to a shorter prefix.
- **Cyclic block masking**: For the iBOT branch a single contiguous block of patches is masked and then cyclically rolled across the patch grid, so the masked region wraps around the borders for better positional coverage.
- **Multi-level objectives and KoLeo**: Like DINOv2, Franca combines the DINO image-level loss, the iBOT patch-level loss, and the KoLeo regularizer [5]_.

Good to Know
------------

- **Building blocks**: Lightly provides the Matryoshka projection head (:class:`~lightly.models.modules.heads.FrancaProjectionHead`), the two matryoshka losses (:class:`~lightly.loss.franca_loss.FrancaDINOLoss` and :class:`~lightly.loss.franca_loss.FrancaIBOTPatchLoss`), and the cyclic block mask (:func:`~lightly.models.utils.random_cyclic_block_mask`). A single-level head or loss reduces to the standard DINOv2 head, DINO loss, and iBOT patch loss.
- **Teacher centering**: Both losses support ``center_mode="mean"`` (a running center, the DINO way) and ``center_mode="sinkhorn"`` (Sinkhorn-Knopp normalization).
- **Relation to other SSL methods**: Franca can be seen as DINOv2 with Matryoshka nested clustering heads and cyclic masking.

Reference:

    .. [0] `Franca: Nested Matryoshka Clustering for Scalable Visual Representation Learning, 2025 <https://arxiv.org/abs/2507.14137>`_
    .. [1] `DINOv2: Learning Robust Visual Features without Supervision, 2023 <https://arxiv.org/abs/2304.07193>`_
    .. [2] `Emerging Properties in Self-Supervised Vision Transformers, 2021 <https://arxiv.org/abs/2104.14294>`_
    .. [3] `iBOT: Image BERT Pre-Training with Online Tokenizer, 2021 <https://arxiv.org/abs/2111.07832>`_
    .. [4] `Matryoshka Representation Learning, 2022 <https://arxiv.org/abs/2205.13147>`_
    .. [5] `Spreading vectors for similarity search, 2018 <https://arxiv.org/abs/1806.03198>`_


.. tabs::
    .. tab:: PyTorch

        .. image:: /_static/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/franca.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/franca.py

        .. literalinclude:: ../../../examples/pytorch/franca.py

    .. tab:: Lightning

        .. image:: /_static/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/franca.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/franca.py

        .. literalinclude:: ../../../examples/pytorch_lightning/franca.py

    .. tab:: Lightning Distributed

        .. image:: /_static/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/franca.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/franca.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm
        - Distributed Sampling is used in the dataloader
        - The matryoshka losses gather their Sinkhorn statistics across processes

        Note that Synchronized Batch Norm is optional and the model can also be
        trained without it. Without Synchronized Batch Norm the batch norm for
        each GPU is only calculated based on the features on that specific GPU.
        Distributed Sampling makes sure that each distributed process sees only
        a subset of the data.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/franca.py
