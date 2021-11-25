.. _models:

Models
======

We provide example implementations for self-supervised learning models 
for PyTorch and PyTorch Lightning to give you a headstart when implementing your own model! 

Note that we adapted the parameters to make the examples easily run 
on a machine with a single GPU. The examples are not optimized for efficiency, 
accuracy or distributed training. Please consult the reference publications f
or the respective models for the optimal settings.

.. toctree::
    :maxdepth: 1
    
    barlowtwins.rst
    byol.rst
    moco.rst
    nnclr.rst
    simclr.rst
    simsiam.rst
    swav.rst