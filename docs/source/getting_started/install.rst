Getting Started 
===================================

Supported Python versions
-------------------------

Lightly requires Python 3.6+. We recommend installing Lighlty in a Linux or OSX environment.

.. _rst-installing:

Installing Lightly
------------------

You can install Lightly and its dependencies from PyPi with:

.. code-block:: bash

    pip install lightly

We strongly recommend that you install Lightly in a dedicated virtualenv, to avoid conflicting with your system packages.

Dependencies
------------
Lightly currently uses `PyTorch <https://pytorch.org/>`_ as the underlying deep learning framework. 
On top of PyTorch we use `Hydra <https://github.com/facebookresearch/hydra>`_ for managing configurations and 
`PyTorch Lightning <https://pytorch-lightning.readthedocs.io/>`_ for training models.

If you want to work with video files you need to additionally install
`PyAV <https://github.com/PyAV-Org/PyAV#installation>`_.

- av

Next Steps
------------

Check out our tutorial: :ref:`lightly-tutorials` 