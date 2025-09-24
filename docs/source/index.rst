.. lightly documentation master file, created by
   sphinx-quickstart on Tue Oct  6 10:38:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: ../logos/lightly_SSL_logo_crop.png
   :width: 600
   :align: center
   :alt: LightlySSL Self-Supervised Learning


Documentation
===================================

.. seealso:: 
   
   These pages document the Lightly self-supervised learning research library. If you
   are instead looking to leverage SSL and distillation pretrain with only a few lines of code
   head over to `LightlyTrain <https://docs.lightly.ai/train/stable/index.html>`_ instead. And if you are looking for the Lightly\ **One** Worker Solution with
   advanced `active learning algorithms <https://docs.lightly.ai/docs/customize-a-selection>`_ and
   `selection strategies <https://docs.lightly.ai/docs/selection-examples-and-use-cases>`_ to select the best samples
   within millions of unlabeled images or video frames stored in your cloud storage or locally,
   please follow our Lightly\ **One** Worker `documentation <https://docs.lightly.ai/>`_.


Lightly\ **SSL** is a computer vision framework for self-supervised learning.

With Lightly\ **SSL** you can train deep learning models using self-supervision. 
This means, that you don’t require any labels to train a model. 
Lightly\ **SSL** has been built to help you understand and work with large unlabeled 
datasets. It is built on top of PyTorch and therefore fully compatible with 
other frameworks such as Fast.ai.

For a commercial version with more features, including Docker support and pretraining
models for embedding, classification, detection, and segmentation tasks with
a single command, please contact sales@lightly.ai.


Lightly AI
----------

.. |lightly_worker_with_bold_one| raw:: html

   <a href="https://docs.lightly.ai" target="_blank">Lightly<strong>One</strong> Worker Solution Documentation</a>


.. |lightly_app_with_bold_one| raw:: html

   <a href="https://app.lightly.ai" target="_blank">Lightly<strong>One</strong> Platform</a>


.. |lightly_train_with_bold_train| raw:: html

   <a href="https://docs.lightly.ai/train/stable/index.html" target="_blank">Lightly<strong>Train</strong> Documentation</a>


- `Homepage <https://www.lightly.ai>`_
- |lightly_train_with_bold_train|
- |lightly_worker_with_bold_one|
- |lightly_app_with_bold_one|
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
   tutorials/package/tutorial_checkpoint_finetuning.rst
   tutorials/package/tutorial_timm_backbone.rst

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
   lightly.models.utils
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
