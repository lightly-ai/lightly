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


There are different levels of synchronization for distributed training. One can
for example just sync the gradients in the backpropagation step. But we can also
sync special layers such as batch norm such that they get the statistics from all
the batches across the GPUs. Some of the additional synchronization might 
improve the final model accuracy at the cost of longer training due to the 
communication overhead.

We did some simple experiments we share here:

Distributed Training Benchmarks
---------------------------

- Dataset: Cifar10 
- Batch size: 512
- Epochs: 100

Distributed training is done with DDP using Pytorch Lightning and the batch size is 
divided by the number of GPUs.

For distributed training we also evaluate whether Synchronized BatchNorm helps and what 
happens if we gather features from all gpus before calculating the 
loss (Gather Distributed).

- Synchronized BatchNorm affects all models
- Gather Distributed only has an effect on SimCLR, BarlowTwins and SwaV.

.. csv-table:: Single GPU
    :header: "Model", "Test Accuracy", "GPUs", "Time", "Peak GPU usage"

    "MoCo",         0.77, 1, "329 min", "11.9 GBytes"
    "SimCLR",       0.79, 1, "208 min", "11.9 GBytes"
    "SimSiam",      0.68, 1, "199 min", "12.0 GBytes"
    "BarlowTwins",  0.64, 1, "197 min", "7.6 GBytes"
    "BYOL",         0.76, 1, "232 min", "7.8 GBytes"
    "SwaV",         0.77, 1, "199 min", "7.5 GBytes"


.. csv-table:: Multi-GPU with Synchronized BatchNorm and Gather Distributed
    :header: "Model", "Test Accuracy", "GPUs", "Time", "Speedup", "Peak GPU usage"

    "MoCo",         0.77, 4, "105 min", 3.13x, "2.2 GBytes"
    "SimCLR",       0.75, 4, "77 min", 2.70x, "2.1 GBytes"
    "SimSiam",      0.67, 4, "79 min", 2.51x, "2.3 GBytes"
    "BarlowTwins",  0.71, 4, "93 min", 2.03x, "2.3 GBytes"
    "BYOL",         0.75, 4, "91 min", 2.55x, "2.3 GBytes"
    "SwaV",         0.77, 4, "78 min", 2.55x, "2.3 GBytes"

.. csv-table:: Multi-GPU with Gather Distributed
    :header: "Model", "Test Accuracy", "GPUs", "Time", "Speedup", "Peak GPU usage"

    "MoCo",         0.76, 4, "89 min", 3.69x, "2.2 GBytes"
    "SimCLR",       0.77, 4, "73 min", 2.75x, "2.1 GBytes"
    "SimSiam",      0.67, 4, "75 min", 2.65x, "2.3 GBytes"
    "BarlowTwins",  0.71, 4, "82 min", 2.40x, "2.3 GBytes"
    "BYOL",         0.76, 4, "91 min", 2.55x, "2.3 GBytes"
    "SwaV",         0.75, 4, "74 min", 2.69x, "2.3 GBytes"


Observations
^^^^^^^^^^^^^^^

- 4 gpus are 2-3x faster than 1 gpu
- With 4 gpus a single epoch takes <40 sec which means that a lot of time is 
  spent between epochs (starting workers, doing evaluation). 
  The benefit from using more gpus could therefore be even greater with a larger dataset.
- The slowdown from Sync BatchNorm is pretty low