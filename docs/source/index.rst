.. lightly documentation master file, created by
   sphinx-quickstart on Tue Oct  6 10:38:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: ../logos/lightly_logo_crop.png
  :width: 600
  :alt: Lightly


Documentation
===================================

Lightly is a computer vision framework for self-supervised learning.

With Lightly you can train deep learning models using self-supervision. 
This means, that you don’t require any labels to train a model. 
Lightly has been built to help you understand and work with large unlabeled 
datasets. It is built on top of PyTorch and therefore fully compatible with 
other frameworks such as Fast.ai.

**NEW** Lightly now has integrated support for active learning in combination 
with the Lightly platform. Use the open-source framework to create embeddings 
of your unlabeled data and combine them with model predictions to select 
the most valuable samples for labeling.
Check it out here: :ref:`lightly-tutorial-active-learning-detectron2`


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/main_concepts.rst
   getting_started/install.rst
   getting_started/command_line_tool.rst
   getting_started/lightly_at_a_glance.rst
   getting_started/active_learning.rst
   getting_started/platform.rst

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   getting_started/advanced.rst
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
   lightly.active_learning
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

   docker/overview.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
