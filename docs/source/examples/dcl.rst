.. _dcl:

DCL & DCLW
==========

Example implementation of the Decoupled Contrastive Learning (DCL) architecture.
DCL is based on the :ref:`simclr` architecture and only introduces a new loss
function. The new loss is called DCL loss and comes also with a weighted form
called DCLW loss. DCL improves upon the widely used NTXent loss (or InfoNCE loss)
by removing a *negative-positive-coupling* effect present in those losses. 
This speeds up model training and allows the usage of smaller batch sizes.

Reference:
    `Decoupled Contrastive Learning, 2021 <https://arxiv.org/abs/2110.06848>`_


DCL is identical to SimCLR but uses :class:`DCLLoss <lightly.loss.dcl_loss.DCLLoss>` 
instead of :class:`NTXentLoss <lightly.loss.ntx_ent_loss.NTXentLoss>`. To use it you can
copy the example code from :ref:`simclr` and make the following adjustments:

.. code::
    
    # instead of this
    from lightly.loss import NTXentLoss
    criterion = NTXentLoss()

    # use this
    from lightly.loss import DCLLoss
    criterion = DCLLoss()

Below you can also find fully runnable examples using the SimCLR architecture
with DCL loss.

.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/dcl.py

        .. literalinclude:: ../../../examples/pytorch/dcl.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/dcl.py

        .. literalinclude:: ../../../examples/pytorch_lightning/dcl.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/dcl.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Synchronized Batch Norm is used in place of standard Batch Norm
        - Features are gathered from all GPUs before the loss is calculated

        Note that Synchronized Batch Norm and feature gathering are optional and
        the model can also be trained without them. Without Synchronized Batch
        Norm and feature gathering the batch norm and loss for each GPU are 
        only calculated based on the features on that specific GPU.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/dcl.py
