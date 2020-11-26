Advanced
===================

In this section, we will have a look at some more advanced topics around lightly. 
For the moment lightly focuses mostly on contrastive learning methods. 
In contrastive learning, we create multiple views of each sample and during 
the training of the model force similar views (originating from the 
same sample) to be close to each other respective different views 
(originating from different samples to be far away. Views are typically 
obtained using augmentation methods.

Through this procedure, we train invariances towards certain augmentations 
when training models using contrastive learning methods. 

Different augmentations result in different invariances. Here we try to group
them and examples on their effect:

**Shape Invariances**

- **Random cropping** --> E.g. We don't care if an object is small or large 
  or only partially in the image

- **Random Horizontal Flip** --> E.g. We don't care about "left and right" in 
  images.

- **Random Vertical Flip** --> E.g. We don't care about "up and down" in images.
  This can be useful for satellite images.

- **Random Rotation** --> E.g. We don't care about the orientation of the camera.
  This can be useful for satellite images.


**Texture Invariances**

- **Gaussian Blur** --> E.g. We don't care about the details of a person but the
  overall shape.


**Color Invariances**

- **Color Jittering** --> E.g. We don't care if a car is blue or red

- **Random Grayscale** --> E.g. We don't care about the color of a tree


Some interesting papers regarding invariances in self-supervised learning:

- `Demystifying Contrastive Self-Supervised Learning, S. Purushwalkam, 2020 <https://arxiv.org/abs/2007.13916>`_
- `What Should Not Be Contrastive in Contrastive Learning, T. Xiao, 2020 <https://arxiv.org/abs/2008.05659>`_


.. note:: Picking the right augmentation method seems crucial for the outcome
          of training models using contrastive learning. For example, if we want
          to create a model classifying cats by color we should not use strong
          color augmentations such as color jittering or random grayscale.


Augmentations
-------------------

Lightly uses the collate operation to apply augmentations when loading a bach 
of samples using the 
`PyTorch dataloader <https://pytorch.org/docs/stable/data.html>`_.

The built-in collate class  
:py:class:`lightly.data.collate.ImageCollateFunction` provides a set of 
common augmentations used in SimCLR and MoCo.

Since Gaussian blur and random rotations by 90 degrees are not supported
by default in torchvision, we added them to lightly 
:py:class:`lightly.transforms`

You can build your own collate function my inheriting from 
:py:class:`lightly.data.collate.BaseCollateFunction`


Models
-------------------

Lightly supports at the moment the following two models for self-supervised
learning:

- `SimCLR: A Simple Framework for Contrastive Learning of Visual Representations, T. Chen, 2020 <https://arxiv.org/abs/2002.05709>`_
  
  - Check the documentation: :py:class:`lightly.models.simclr.ResNetSimCLR`

- `MoCo: Momentum Contrast for Unsupervised Visual Representation Learning, K. He, 2019 <https://arxiv.org/abs/1911.05722>`_
  
  - Check the documentation: :py:class:`lightly.models.moco.ResNetMoCo`

Do you know a model that should be on this list? Please add an issue on GitHub :)



Losses 
-------------------

We provide the most common loss fucntion for contrastive learning. 

- `NTXentLoss: Normalized Temperature-scaled Cross Entropy Loss <https://paperswithcode.com/method/nt-xent>`_

  - Check the documentation: :py:class:`lightly.loss.ntx_ent_loss.NTXentLoss`


Memory Bank
^^^^^^^^^^^^^^^^^^^

Since contrastive learning methods benefit from many negative examples larger
batch sizes are preferred. However, not everyone has a multi GPU cluster at 
hand. Therefore, alternative tricks and methods have been derived in research.
On of them is a memory bank keeping past examples as additional negatives.

For an example of the memory bank in action have a look at 
:ref:`lightly-moco-tutorial-2`. 

For more information check the documentation: 
:py:class:`lightly.loss.memory_bank.MemoryBankModule`.

