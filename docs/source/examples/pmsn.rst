.. _pmsn:

PMSN
====

Prior Matching for Siamese Networks (PMSN) builds on top of :ref:`MSN` by adding support
for custom clustering priors. This is especially helpful for datasets with non-uniform
class distributions. By default, PMSN uses a power law distribution which is ideal
for datasets with long tail distributions.

Reference:
    `The Hidden Uniform Cluster Prior in Self-Supervised Learning, 2022 <https://arxiv.org/abs/2210.07277>`_


For PMSN, you can use the exact same code as for :ref:`msn` but change the 
target distribution in :py:class:`lightly.loss.msn_loss.MSNLoss` to :code:`"power_law"`:

.. code::

    # instead of this
    from lightly.loss import MSNLoss
    criterion = MSNLoss()

    # use this
    criterion = MSNLoss(target_distribution="power_law", power_law_exponent=0.25)

    # or define your custom target distribution
    def my_uniform_target_distribution(mean_anchor_probabilities: Tensor) -> Tensor:
        dim = mean_anchor_probabilities.shape[0]
        return mean_anchor_probabilities.new_ones(dim) / dim

    criterion = MSNLoss(target_distribution=my_uniform_target_distribution)


.. tabs::
    .. tab:: PyTorch

        This example can be run from the command line with::

            python lightly/examples/pytorch/pmsn.py

        .. literalinclude:: ../../../examples/pytorch/pmsn.py

    .. tab:: Lightning

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/pmsn.py

        .. literalinclude:: ../../../examples/pytorch_lightning/pmsn.py

    .. tab:: Lightning Distributed

        This example runs on multiple gpus using Distributed Data Parallel (DDP)
        training with Pytorch Lightning. At least one GPU must be available on 
        the system. The example can be run from the command line with::

            python lightly/examples/pytorch_lightning_distributed/pmsn.py

        The model differs in the following ways from the non-distributed
        implementation:

        - Distributed Data Parallel is enabled
        - Distributed Sampling is used in the dataloader
        - Distributed Sinkhorn is used in the loss calculation 

        Distributed Sampling makes sure that each distributed process sees only
        a subset of the data.

        .. literalinclude:: ../../../examples/pytorch_lightning_distributed/pmsn.py
