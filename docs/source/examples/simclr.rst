.. _simclr:

SimCLR
======

Example implementation of the SimCLR architecture.

Reference:
    `A Simple Framework for Contrastive Learning of Visual Representations, 2020 <https://arxiv.org/abs/2002.05709>`_

Tutorials:
    :ref:`lightly-simclr-tutorial-3`


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/simclr.py

        .. literalinclude:: ../../../examples/pytorch/simclr.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/simclr.py

        .. literalinclude:: ../../../examples/pytorch_lightning/simclr.py


.. note::
    We adapted the parameters to make the example easily run on a machine with a single GPU.
    Please consult the paper if you want to follow the original training settings.
