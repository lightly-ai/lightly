.. _frossl:

FroSSL
======

FroSSL (Frobenius norm minimization for Self-Supervised Learning) is a method that
avoids representation collapse by maximizing the entropy of each view's
(trace-normalized) covariance matrix through its squared Frobenius norm, while an
invariance term pulls the views towards their mean. It uses a VICReg/Barlow Twins
style projection head and, unlike two-view objectives, naturally supports any number
of views.

Reference:
    `FroSSL: Frobenius Norm Minimization for Self-Supervised Learning, 2024 <https://arxiv.org/abs/2310.02903>`_


.. tabs::

    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/frossl.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/frossl.py

        .. literalinclude:: ../../../examples/pytorch/frossl.py

    .. tab:: Lightning

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/frossl.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/frossl.py

        .. literalinclude:: ../../../examples/pytorch_lightning/frossl.py
