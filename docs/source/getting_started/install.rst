Installation
===================================

Supported Python versions
-------------------------

Lightly\ **SSL** requires Python 3.7+. We recommend installing Lightly\ **SSL** in a Linux or OSX environment.

.. _rst-installing:

Installing LightlySSL
----------------------

You can install Lightly\ **SSL** and its dependencies from PyPi with:

.. code-block:: bash

    pip install lightly

We strongly recommend that you install Lightly\ **SSL** in a dedicated virtualenv, to avoid conflicting with your system packages.

Dependencies
------------
Lightly\ **SSL** currently uses `PyTorch <https://pytorch.org/>`_ as the underlying deep learning framework. 
On top of PyTorch we use `Hydra <https://github.com/facebookresearch/hydra>`_ for managing configurations and 
`PyTorch Lightning <https://pytorch-lightning.readthedocs.io/>`_ for training models.

If you want to work with video files you need to additionally install
`PyAV <https://github.com/PyAV-Org/PyAV#installation>`_.

.. code-block:: bash

    pip install "lightly[video]"

If you want to work with the Masked Autoencoder you need to additionally install
`TIMM <https://github.com/huggingface/pytorch-image-models>`_.

.. code-block:: bash

    pip install "lightly[timm]"

Next Steps
------------

Start with one of our tutorials: :ref:`input-structure-label`
