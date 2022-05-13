.. _ref-hardware-recommendations:

Hardware recommendations
========================

Usually, the Lightly worker is run on a cloud machine,
which is spin up specifically to run the docker.
Our recommendations on the hardware configuration of this machine are
based on three criteria:

- speed: The worker should process your dataset as quickly a possible.
- cost-effectiveness: The machine should be cheap to rent.
- stability: The worker should not crash because it runs out of memory.

As a starting point, we recommend a machine with the following configuration:

- 8 vCPUs
- 30-32 GB of system memory
- a NVIDIA-GPU with 16GB of video memory, e.g. the T4 GPU.

If you read the data from a local hard disk, make sure that the connection is fast,
ideally faster than 100MB/s.

If you read the data from a cloud bucket instead, make sure that
the cloud bucket is in the same region as the compute machine
and the download speed is fast, ideally faster than 100MB/s.

Keep the configuration option `lightly.loader.num_workers` at the default (-1),
which will set it to the number of vCPUs on your machine.

The best machine hardware configuration is highly dependent
on the type of data you have and how you configure the worker.
Thus we recommend to adapt it to your needs.

Finding the compute speed bottleneck
------------------------------------

Usually, the compute speed is limited by one of three potential bottlenecks:

- data read speed: I/O
- CPU
- GPU

Different steps of the Lightly worker use these resources to a different extent.
Thus the bottleneck changes throughout the run.
The GPU is used at three steps:

- training an embedding model (optional step)
- pretagging your dataset (optional step)
- embedding your dataset

The I/O and CPUs are used at the former 3 steps and also used at the other steps that may take longer:

- initializing the dataset
- corruptness check (optional step)
- dataset dumping & upload (optional step)

Before updating one or all of resources, we recommend finding out the current bottleneck:

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


