.. _byol:

BYOL
====

Example implementation of the BYOL architecture.

Reference:
    `Bootstrap your own latent: A new approach to self-supervised Learning, 2020 <https://arxiv.org/abs/2006.07733>`_


.. tabs::

    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/byol.py

        .. literalinclude:: ../../../examples/pytorch/byol.py

    .. tab:: Lighning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/byol.py

        .. literalinclude:: ../../../examples/pytorch_lightning/byol.py


.. note::
    We adapted the parameters to make the example easily run on a machine with a single GPU.
    Please consult the paper if you want to follow the original training settings.
