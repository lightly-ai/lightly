.. _nnclr:

NNCLR
=====

Example implementation of the NNCLR architecture.

Reference:
    `With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations, 2021 <https://arxiv.org/abs/2104.14548>`_

.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/nnclr.py

        .. literalinclude:: ../../../examples/pytorch/nnclr.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/nnclr.py

        .. literalinclude:: ../../../examples/pytorch_lightning/nnclr.py


.. note::
    We adapted the parameters to make the example easily run on a machine with a single GPU.
    Please consult the paper if you want to follow the original training settings.
