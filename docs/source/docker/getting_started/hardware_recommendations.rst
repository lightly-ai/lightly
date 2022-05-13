.. _ref-hardware-recommendations:

Hardware recommendations
========================

Lightly worker is usually run on dedicated hardware
or in the cloud on a compute instance
which is specifically spun up to run Lightly Worker standalone.
Our recommendations on the hardware requirements of this compute instance are
based on three criteria:

- speed: The worker should process your dataset as quickly a possible.
- cost-effectiveness: The compute instance should be economical.
- stability: The worker should not crash because it runs out of memory.

Depending on your dataset size, we recommend the following machine:

- Up to 100.000 images or video frames: Use the AWS EC2 instance `g4dn.xlarge` or similar
  with 4 vCPUs, 16GB of system memory, one T4 GPU
- Up to 1 Million images or video frames: Use the AWS EC2 instance `g4dn.2xlarge` or similar
  with 8 vCPUs, 32GB of system memory, one T4 GPU
- More than 1 Million images or video frames: Use the AWS EC2 instance `g4dn.4xlarge` or similar
  with 16 vCPUs, 64GB of system memory, one T4 GPU

You can compute the number of frames of your videos with their length and fps.
E.g. 100 videos with 600s length each and 30 fps have 100 * 600 * 30 = 1.8 Mio frames.

If you want to train an embedding model for many epochs or want to further increase computing speed,
we recommend to switch to a V100 or A10 GPU or better.

If you stream the data from a cloud bucket using the datasource feature, make sure that
the cloud bucket is in the same region as the compute machine.
Using the same region is very important, see also :ref:`ref-docker-network-traffic-same-region`
If you are using the old workflow of reading from a local disk instead, use a SSD.
However, we recommend the workflow to stream from a cloud bucket.


Keep the configuration option `lightly.loader.num_workers` at the default (-1),
which will set it to the number of vCPUs on your machine.

Finding the compute speed bottleneck
------------------------------------

Usually, the compute speed is limited by one of three potential bottlenecks.
Different steps of the Lightly worker use these resources to a different extent.
Thus the bottleneck changes throughout the run. The bottlenecks are:

- data read speed: I/O
- CPU
- GPU


The GPU is used during three steps:

- training an embedding model (optional step)
- pretagging your dataset (optional step)
- embedding your dataset

The I/O and CPUs are used during the previous 3 steps and also used during the other steps that may take longer:

- initializing the dataset
- corruptness check (optional step)
- dataset dumping & upload (optional step)

Before changing the hardware configuration of your compute instance,
we recommend to first determine the bottleneck by monitoring it:

- You can find out the current disk usage of your machine using the terminal command `iotop`.
- If you use a datasource, see the current ethernet usage using the terminal command `ifstat`.
- You can find out the current CPU and RAM usage of your machine using the terminal commands `top` or `htop`.
- You can find out the current GPU usage (both compute and VRAM) using the terminal command `watch nvidia-smi`.
- Note that you might need to install these commands using your package manager.


Additional to using these tools, you can also compare the relative duration of the different steps to see the bottleneck.
E.g. if the embedding step takes much longer than the corruptness check, then the GPU is the bottleneck.
Otherwise, it is the I/O or CPU.

Updating the machine
--------------------

When updating the machine, we recommend updating the resource that causes the
bottleneck. After that, the bottleneck might have changed.

If there is not one obvious bottleneck, we recommend to scale up I/O, CPUs and GPU together.

To prevent the worker running out of system memory or GPU memory, we recommend
about 4GB of RAM and 2GB ov VRAM for each vCPU.


