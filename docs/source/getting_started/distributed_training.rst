.. _lightly-distributed-training:

Distributed Training
====================

Lightly supports training your model on multiple GPUs using Pytorch Lightning
and Distributed Data Parallel (DDP) training. You can find reference
implementations for all our models in the :ref:`models` section.

Training with multiple gpus is also available from the command line: :ref:`cli-train-lightly`

For details on distributed training we recommend the following pages:

- `Pytorch Distributed Overview <https://pytorch.org/tutorials/beginner/dist_overview.html>`_
- `Pytorch Lightning Multi-GPU Training <https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html>`_