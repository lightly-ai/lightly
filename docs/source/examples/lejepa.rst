.. _lejepa:

LeJEPA
======

LeJEPA [0]_ is a self-supervised learning method that learns image
representations by enforcing invariance between multiple augmented views of
the same image while regularizing the projected embeddings with SIGReg
(Sketched Isotropic Gaussian Regularization). The objective combines a
simple mean-squared invariance term with a SIGReg penalty that drives the
projected features toward an isotropic Gaussian, removing the need for
negative samples, momentum encoders, or stop-gradients.

Key Components
--------------

- **Multi-view projections**: LeJEPA uses a shared backbone and projection
  head to produce embeddings for several global views and a set of local
  (smaller) views.
- **Invariance loss**: Each local view's projection is pulled toward the
  centroid of the global views' projections using a mean-squared distance.
- **SIGReg**: A sliced, Epps-Pulley based regularizer that compares the
  empirical characteristic function of the projected embeddings to that of
  an isotropic Gaussian. It is the only regularizer in the LeJEPA objective.
- **Projection head**: A multi-layer perceptron with BatchNorm and ReLU
  (:class:`lightly.models.modules.LeJEPAProjectionHead`) maps backbone
  features into the projection space where the loss is computed.

Good to Know
------------

- **No negatives or momentum encoder**: Unlike contrastive (SimCLR, MoCo) or
  self-distillation (DINO, BYOL) methods, LeJEPA only needs positive views
  and a shared online encoder.
- **SIGReg replaces VICReg-style variance/covariance terms**: The Gaussian
  matching objective is more principled and avoids the heuristic
  hyperparameters of variance and covariance regularizers.
- **Compatible with DINO-style multi-crop transforms**: The example uses
  :class:`lightly.transforms.DINOTransform` with global and local crops,
  matching the LeJEPA paper's training setup.

Reference:

    .. [0] `LeJEPA, 2025 <https://arxiv.org/abs/2511.08544>`_
    .. [1] `LeJEPA reference implementation <https://github.com/galilai-group/lejepa>`_

.. note::

    LeJEPA requires `TIMM <https://github.com/huggingface/pytorch-image-models>`_ to be
    installed

    .. code-block:: bash

        pip install "lightly[timm]"

.. tabs::
    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/lejepa.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/lejepa.py

        .. literalinclude:: ../../../examples/pytorch/lejepa.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/lejepa.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/lejepa.py

        .. literalinclude:: ../../../examples/pytorch_lightning/lejepa.py

    .. tab:: Lightning Distributed

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/lejepa.ipynb

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/lejepa.py

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/lejepa.py
