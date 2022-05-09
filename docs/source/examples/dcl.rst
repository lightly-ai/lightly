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

    # or alternatively the weighted DCL loss
    from lightly.loss import DCLWLoss
    criterion = DCLWLoss()

    # for distributed training you can enable gather_distributed to calculate
    # the loss over the all distributed samples
    from lightly.loss import DCLLoss
    criterion = DCLLoss(gather_distributed=True)
