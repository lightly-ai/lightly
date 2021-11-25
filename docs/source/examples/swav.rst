.. _swav:

SwaV
====

Example implementation of the SwaV architecture.

Reference:
    `Unsupervised Learning of Visual Features by Contrasting Cluster Assignments, 2020 <https://arxiv.org/abs/2006.09882>`_


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/swav.py

        .. literalinclude:: ../../../examples/pytorch/swav.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/swav.py

        .. literalinclude:: ../../../examples/pytorch_lightning/swav.py


.. note::
    We adapted the parameters to make the example easily run on a machine with a single GPU.
    Please consult the paper if you want to follow the original training settings.
