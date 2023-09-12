.. _lightly-main-concepts:

Main Concepts
=============

Self-Supervised Learning
------------------------

The figure below shows an overview of the different concepts used by the Lightly SSL package
and a schema of how they interact. The expressions in **bold** are explained further
below.

.. figure:: images/lightly_overview.png
    :align: center
    :alt: Lightly SSL Overview

    Overview of the different concepts used by the Lightly SSL package and how they interact.

* **Dataset**
   In Lightly SSL, datasets are accessed through :py:class:`~lightly.data.dataset.LightlyDataset`.
   You can create a :py:class:`~lightly.data.dataset.LightlyDataset` from a directory of
   images or videos, or directly from a `torchvision dataset <https://pytorch.org/vision/stable/datasets.html>`_.
   You can learn more about this in our tutorial: 

   * :ref:`input-structure-label`

* **Transform**
   In self-supervised learning, the input images are often randomly transformed into
   *views* of the orignal images. The views and their underlying transforms are
   important as they define the properties of the model and the image embeddings.
   You can either use our pre-defined :py:mod:`~lightly.transforms` or write your own.
   For more information, check out the following pages:

   * :ref:`lightly-advanced`
   * :ref:`lightly-custom-augmentation-5`.

* **Collate Function**
   The collate function aggregates the views of multiple images into a single batch.
   You can use the default collate function. Lightly SSL also provides a  
   :py:class:`~lightly.data.multi_view_collate.MultiViewCollate`

* **Dataloader**
   For the dataloader you can simply use a `PyTorch dataloader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.
   Be sure to pass it a :py:class:`~lightly.data.dataset.LightlyDataset` though!

* **Backbone Neural Network**
   One of the cool things about self-supervised learning is that you can pre-train
   your neural networks without the need for annotated data. You can plugin whatever
   backbone you want! If you don't know where to start, have a look at our :ref:`simclr`
   example on how to use a `ResNet <https://pytorch.org/vision/main/models/resnet.html>`_ 
   backbone or :ref:`msn` for a `Vision Transformer <https://pytorch.org/vision/main/models/vision_transformer.html>`_
   backbone.

* **Heads**
   The heads are the last layers of the neural network and added on top of the backbone.
   They project the outputs of the backbone, commonly called *embeddings*,
   *representations*, or *features*, into a new space in which the loss is calculated.
   This has been found to be hugely beneficial instead of directly calculating the loss
   on the embeddings. Lightly SSL provides common :py:mod:`~lightly.models.modules.heads`
   that can be added to any backbone.

* **Model**
   The model combines your backbone neural network with one or multiple heads and, if
   required, a momentum encoder to provide an easy-to-use interface to the most
   popular self-supervised learning models. Our :ref:`models <models>` page contains
   a large number of example implementations. You can also head over to one of our
   tutorials if you want to learn more about models and how to use them:

   * :ref:`sphx_glr_tutorials_package_tutorial_moco_memory_bank.py`
   * :ref:`sphx_glr_tutorials_package_tutorial_simclr_clothing.py`
   * :ref:`sphx_glr_tutorials_package_tutorial_simsiam_esa.py`

* **Loss**
   The loss function plays a crucial role in self-supervised learning. Lightly SSL provides
   common loss functions in the :py:mod:`~lightly.loss` module.

* **Optimizer**
   With Lightly SSL, you can use any `PyTorch optimizer <https://pytorch.org/docs/stable/optim.html>`_
   to train your model.

* **Training**
   The model can either be trained using a plain `PyTorch training loop <https://pytorch.org/tutorials/beginner/introyt/trainingyt.html>`_
   or with a dedicated framework such as `PyTorch Lightning <https://www.pytorchlightning.ai/index.html>`_.
   Lightly SSL lets you choose what is best for you. Check out our :ref:`models <models>` and
   `tutorials <https://docs.lightly.ai/self-supervised-learning/tutorials/package.html>`_
   sections on how to train models with PyTorch or PyTorch Lightning.

* **Image Embeddings**
   During the training process, the model learns to create compact embeddings from images.
   The embeddings, also often called representations or features, can then be used for
   tasks such as identifying similar images or creating a diverse subset from your data:

   * :ref:`lightly-tutorial-sunflowers`
   * :ref:`lightly-simsiam-tutorial-4`

* **Pre-Trained Backbone**
   The backbone can be reused after self-supervised training. It can be transferred to
   any other task that requires a similar network architecture, including
   image classification, object detection, and segmentation tasks. You can learn more in
   our object detection tutorial:

   * :ref:`lightly-detectron-tutorial-6`
