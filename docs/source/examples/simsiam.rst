.. _simsiam:

SimSiam
=======

Example implementation of the SimSiam architecture.

Reference:
    `Exploring Simple Siamese Representation Learning, 2020 <https://arxiv.org/abs/2011.10566>`_

Tutorials:
    :ref:`lightly-simsiam-tutorial-4`


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/simsiam.py

        .. literalinclude:: ../../../examples/pytorch/simsiam.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/simsiam.py

        .. literalinclude:: ../../../examples/pytorch_lightning/simsiam.py


.. note::
    We adapted the parameters to make the example easily run on a machine with a single GPU.
    Please consult the paper if you want to follow the original training settings.
