.. _barlowtwins:


Barlow Twins
============

Example implementation of the Barlow Twins architecture.

Reference:
    `Barlow Twins: Self-Supervised Learning via Redundancy Reduction, 2021 <https://arxiv.org/abs/2103.03230>`_

This example can be run from the command line with::

    python lightly/examples/barlowtwins.py

.. literalinclude:: ../../../lightly/examples/barlowtwins.py

.. note::
    We adapted the parameters to make the example easily run on a machine with a single GPU.
    Please consult the paper if you want to follow the original training settings.