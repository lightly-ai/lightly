.. lightly documentation master file, created by
   sphinx-quickstart on Tue Oct  6 10:38:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: ../logos/lightly_logo_crop.png
  :width: 600
  :alt: Lightly


Documentation
===================================

.. note:: These pages document the Lightly self-supervised learning library.
          If you are looking for Lightly Worker Solution to easily process millions
          of samples and run powerful active learning algorithms on your data
          please follow
          `Lightly Worker documentation <https://docs.lightly.ai/>`_.

Lightly is a computer vision framework for self-supervised learning.

With Lightly you can train deep learning models using self-supervision. 
This means, that you don’t require any labels to train a model. 
Lightly has been built to help you understand and work with large unlabeled 
datasets. It is built on top of PyTorch and therefore fully compatible with 
other frameworks such as Fast.ai.


Lightly
-------

- `Homepage <https://www.lightly.ai>`_
- `Web-App <https://app.lightly.ai>`_
- `Documentation <https://docs.lightly.ai/self-supervised-learning/>`_
- `Lightly Solution Documentation (Lightly Worker & API) <https://docs.lightly.ai/>`_
- `Github <https://github.com/lightly-ai/lightly>`_
- `Discord <https://discord.gg/xvNJW94>`_ (We have weekly paper sessions!)

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/main_concepts.rst
   getting_started/install.rst
   getting_started/command_line_tool.rst
   getting_started/lightly_at_a_glance.rst

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   getting_started/advanced.rst
   getting_started/distributed_training.rst
   getting_started/benchmarks.rst

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/package.rst
   tutorials/platform.rst

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/models.rst

.. toctree::
   :maxdepth: 1
   :caption: Python API

   lightly
   lightly.api
   lightly.cli
   lightly.core
   lightly.data
   lightly.loss
   lightly.models
   lightly.transforms
   lightly.utils


.. toctree::
   :maxdepth: 1
   :caption: On-Premise

   docker_archive/overview.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
