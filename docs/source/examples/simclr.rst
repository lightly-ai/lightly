.. _simclr:

SimCLR
======

SimCLR is a self-supervised framework for visual representation learning using contrastive methods. It learns by creating two augmented views of the same image—using random cropping, color jitter, and Gaussian blur—then maximizing agreement between these augmented views while separating them from other images. Key findings include the importance of strong compositions of data augmentations, a nonlinear projection head that boosts representation quality, and the advantages of large batch sizes. Combined, these elements allow SimCLR to approach or match supervised performance on ImageNet and achieve strong transfer and semi-supervised learning results.

Key Components
--------------

- **Data Augmentations**: SimCLR uses random cropping, resizing, color jittering, and Gaussian blur to create diverse views of the same image.
- **Backbone**: Convolutional neural networks, such as ResNet, are employed to encode augmented images into feature representations.
- **Projection Head**: A multilayer perceptron (MLP) maps features into a space where contrastive loss is applied, enhancing representation quality.
- **Contrastive Loss**: The normalized temperature-scaled cross-entropy loss (NT-Xent) encourages similar pairs to align and dissimilar pairs to diverge.

Good to Know
----------------

- **Backbone Networks**: SimCLR is specifically optimized for convolutional neural networks, with a focus on ResNet architectures. We do not recommend using it with transformer-based models.
- **Learning Paradigm**: SimCLR is based on contrastive learning which makes it sensitive to the augmentations you pick and the method benefits from larger batch sizes.

Reference:
    `A Simple Framework for Contrastive Learning of Visual Representations, 2020 <https://arxiv.org/abs/2002.05709>`_

Tutorials:
    :ref:`lightly-simclr-tutorial-3`

The example
-----------

One file, plain PyTorch, with the training loop in view. Run it with::

    python examples/simclr.py

.. literalinclude:: ../../../examples/simclr.py

Reproducing a published number
------------------------------

``benchmarks/simclr/`` runs the same method on Lightning, with the paper's
settings, the probes and DDP. It carries one row per dataset, and the example
above is the small row written out::

    torchrun --nproc_per_node=8 -m benchmarks.simclr.benchmark \
        --train-dir /datasets/imagenet/train --val-dir /datasets/imagenet/val

The two files restate each other rather than one importing the other, and
``tests/test_simclr_agrees.py`` is what holds them to the same method. They do
not share numbers: the example is small enough to run on one GPU, the benchmark
is the paper.

For configured training on your own data with any backbone, use
`LightlyTrain <https://docs.lightly.ai/train/stable/>`_.
