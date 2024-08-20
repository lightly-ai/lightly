.. _pmsn:

PMSN
====

Prior Matching for Siamese Networks (PMSN) builds on top of :ref:`MSN` by adding support
for custom clustering priors. This is especially helpful for datasets with non-uniform
class distributions. By default, PMSN uses a power law distribution which is ideal
for datasets with long tail distributions.

Reference:
    `The Hidden Uniform Cluster Prior in Self-Supervised Learning, 2022 <https://arxiv.org/abs/2210.07277>`_


For PMSN, you can use the exact same code as for :ref:`msn` but change
:py:class:`lightly.loss.msn_loss.MSNLoss` to :py:class:`lightly.loss.pmsn_loss.PMSNLoss`:

.. code::

    # instead of this
    from lightly.loss import MSNLoss
    criterion = MSNLoss()

    # use this
    from lightly.loss import PMSNLoss
    criterion = PMSNLoss(power_law_exponent=0.25)

    # or define your custom target distribution
    from lightly.loss import PMSNCustomLoss
    def my_uniform_target_distribution(mean_anchor_probabilities: Tensor) -> Tensor:
        dim = mean_anchor_probabilities.shape[0]
        return mean_anchor_probabilities.new_ones(dim) / dim

    criterion = PMSNCustomLoss(target_distribution=my_uniform_target_distribution)


.. tabs::
    .. tab:: PyTorch

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/pmsn.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch/pmsn.py

        .. literalinclude:: ../../../examples/pytorch/pmsn.py

    .. tab:: Lightning

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/pmsn.ipynb

        This example can be run from the command line with::

            python lightly/examples/pytorch_lightning/pmsn.py

        .. literalinclude:: ../../../examples/pytorch_lightning/pmsn.py

    .. tab:: Lightning Distributed

        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning_distributed/pmsn.ipynb

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
