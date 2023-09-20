.. lightly documentation master file, created by
   sphinx-quickstart on Tue Oct  6 10:38:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: ../logos/lightly_SSL_logo_crop.png
   :width: 600
   :align: center
   :alt: Lightly SSL Self-Supervised Learning


Documentation
===================================

.. note:: These pages document the Lightly self-supervised learning library.
          If you are looking for the Lightly Worker Solution with
          advanced `active learning algorithms <https://docs.lightly.ai/docs/selection>`_ and
          `selection strategies <https://docs.lightly.ai/docs/selection>`_ to select the best samples
          within millions of unlabeled images or video frames stored in your cloud storage or locally,
          please follow our `Lightly Worker documentation <https://docs.lightly.ai/>`_.

Lightly SSL is a computer vision framework for self-supervised learning.

With Lightly SSL you can train deep learning models using self-supervision. 
This means, that you don’t require any labels to train a model. 
Lightly SSL has been built to help you understand and work with large unlabeled 
datasets. It is built on top of PyTorch and therefore fully compatible with 
other frameworks such as Fast.ai.


Lightly AI
----------

- `Homepage <https://www.lightly.ai>`_
- `Lightly Worker Solution Documentation <https://docs.lightly.ai/>`_
- `Lightly Platform <https://app.lightly.ai>`_
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

   tutorials/structure_your_input.rst
   tutorials/package/tutorial_moco_memory_bank.rst
   tutorials/package/tutorial_simclr_clothing.rst
   tutorials/package/tutorial_simsiam_esa.rst
   tutorials/package/tutorial_custom_augmentations.rst
   tutorials/package/tutorial_pretrain_detectron2.rst

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
