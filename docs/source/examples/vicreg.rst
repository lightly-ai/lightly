.. _vicreg:

 VICReg
 ===

 VICReg (Variance-Invariance-Covariance Regularization) is a method that explicitly
 avoids the collapse problem with a simple regularization term on the variance of the embeddings along each dimension individually. VICReg combines the
 variance term with a decorrelation mechanism based on redundancy reduction and covariance regularization. It inherits the model structure from 
 `Barlow Twins, 2022 <https://arxiv.org/abs/2103.03230>`_ changing the loss. Doing so allows the stabilization of the training and leads to performance improvements. 

 Reference:
     `VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning, 2022 <https://arxiv.org/abs/2105.04906>`_


 .. tabs::
     .. tab:: PyTorch

         This example can be run from the command line with::

             python lightly/examples/pytorch/vicreg.py

         .. literalinclude:: ../../../examples/pytorch/vicreg.py

     .. tab:: Lightning

         This example can be run from the command line with::

             python lightly/examples/pytorch_lightning/vicreg.py

         .. literalinclude:: ../../../examples/pytorch_lightning/vicreg.py

     .. tab:: Lightning Distributed

         This example runs on multiple gpus using Distributed Data Parallel (DDP)
         training with Pytorch Lightning. At least one GPU must be available on 
         the system. The example can be run from the command line with::

             python lightly/examples/pytorch_lightning_distributed/vicreg.py

         The model differs in the following ways from the non-distributed
         implementation:

         - Distributed Data Parallel is enabled
         - Distributed Sampling is used in the dataloader
         - Distributed Sinkhorn is used in the loss calculation 

         Distributed Sampling makes sure that each distributed process sees only
         a subset of the data.

         .. literalinclude:: ../../../examples/pytorch_lightning_distributed/vicreg.py