.. _moco:

MoCo
====

Example implementation of the MoCo v2 architecture.

References:
    MoCo v1: `Momentum Contrast for Unsupervised Visual Representation Learning, 2020 <https://arxiv.org/abs/1911.05722v3>`_

    MoCo v2: `Improved Baselines with Momentum Contrastive Learning, 2020 <https://arxiv.org/abs/2003.04297>`_

    MoCo v3: `An Empirical Study of Training Self-Supervised Vision Transformers, 2021 <https://arxiv.org/abs/2104.02057>`_

Tutorials:
    :ref:`lightly-moco-tutorial-2`


.. tabs::

    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/moco.py

        .. literalinclude:: ../../../examples/pytorch/moco.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/moco.py

        .. literalinclude:: ../../../examples/pytorch_lightning/moco.py

