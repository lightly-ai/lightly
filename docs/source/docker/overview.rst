Docker
===================================

We all know that sometimes when working with ML data we deal with really BIG datasets. The cloud solution is great for exploration, prototyping
and an easy way to work with lightly. But there is more!

With the introduction of our on-premise solution you can process larger datasets completely on your end without data leaving your infrastructure.
We worked hard to make this happen and are very proud to present you the following specs:

* Sample more than 1 Million samples within a few hours!

* Wrapped in a docker container (no setup required if your system supports docker)

* Configurable

  * Use stopping conditions for sampling such as minimum distance between
    two samples

  * Use various sampling methos

  * Check for corrupt files and report them

  * Check for exact duplicates and report them

  * We expose the full lightly framework config

* Automated reporting of the datasets for each run

  * PDF report with histograms, plots, statistics and much more ...

* Hand-optimized code (to instruction level)

  * Multithreaded

  * SIMD instructions

* Minimal hardware requirements:

  * 1 CPU core

  * 4 GB free RAM

* Recommended hardware:
  
  * 8 CPU cores or more

  * 16GB free RAM
 
  * 1 Nvidia Tesla P100 or V100 GPU with CUDA 10.0+


.. toctree::
   :maxdepth: 1

   getting_started/setup.rst
   getting_started/first_steps.rst
   configuration/configuration.rst

Examples 
-----------------------------------

.. toctree::
   :maxdepth: 1

   examples/imagenet.rst


Changelog
-------------

**13.11.2020**
 * Supports training, embedding, sampling
 * Filter corrupt images
 * Remove exact duplicates