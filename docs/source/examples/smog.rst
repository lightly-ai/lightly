.. _smog:

SMoG
====

Example implementation of the Synchronous Momentum Grouping (SMoG) paper. 
SMoG follows the framework of contrastive learning but replaces the contrastive
unit from instance to group, mimicking clustering-based methods. To
achieve this, they propose the momentum grouping scheme which synchronously 
conducts feature grouping with representation learning. 

Reference:
    `Unsupervised Visual Representation Learning by Synchronous Momentum Grouping, 2022 <https://arxiv.org/pdf/2207.06167.pdf>`_


.. tabs::
    .. tab:: PyTorch

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/smog.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/smog.py

        .. literalinclude:: ../../../examples/pytorch/smog.py

    .. tab:: Lightning

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/smog.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/smog.py

        .. literalinclude:: ../../../examples/pytorch_lightning/smog.py
