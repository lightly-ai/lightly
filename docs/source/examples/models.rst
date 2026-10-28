.. _models:

Models
======

Self-supervised methods with runnable examples in PyTorch, PyTorch Lightning and
distributed PyTorch Lightning. Each one is built from the same losses, heads and
transforms, so you can start from a method and swap the parts.

.. tip::

   Not sure where to start? With a ResNet, try :ref:`simclr` or :ref:`byol`. With a
   ViT, try :ref:`dinov2` or :ref:`mae`.

.. raw:: html

   <div class="ls-filter" role="group" aria-label="Filter methods by family">
     <button type="button" class="ls-filter-btn is-active" data-family="all">All</button>
     <button type="button" class="ls-filter-btn" data-family="contrastive">Contrastive</button>
     <button type="button" class="ls-filter-btn" data-family="distillation">Self-distillation</button>
     <button type="button" class="ls-filter-btn" data-family="masked">Masked image modeling</button>
     <button type="button" class="ls-filter-btn" data-family="redundancy">Redundancy reduction</button>
     <button type="button" class="ls-filter-btn" data-family="clustering">Clustering and dense</button>
     <button type="button" class="ls-filter-btn" data-family="world">World models</button>
   </div>

.. Each card's column carries ls-family-<key>; _static/method-filter.js hides the
   columns of the other families and fills in the count on each button.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: ls-filter-target

   .. grid-item-card:: SimCLR
      :link: simclr
      :link-type: ref
      :class-item: ls-fam-item ls-family-contrastive
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Contrastive · 2020

      Pulls two augmented views of an image together and pushes other images away.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span><span>Tutorial</span></p>

   .. grid-item-card:: MoCo
      :link: moco
      :link-type: ref
      :class-item: ls-fam-item ls-family-contrastive
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Contrastive · 2020

      Contrastive learning with a momentum encoder and a memory bank of negatives.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span><span>Tutorial</span></p>

   .. grid-item-card:: NNCLR
      :link: nnclr
      :link-type: ref
      :class-item: ls-fam-item ls-family-contrastive
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Contrastive · 2021

      Uses a view's nearest neighbour from a support set as its positive.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: DCL & DCLW
      :link: dcl
      :link-type: ref
      :class-item: ls-fam-item ls-family-contrastive
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Contrastive · 2021

      SimCLR with a decoupled loss that drops the positive pair from the denominator.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: TiCo
      :link: tico
      :link-type: ref
      :class-item: ls-fam-item ls-family-contrastive
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Contrastive · 2022

      Agreement between distorted views plus a covariance contrast term.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: BYOL
      :link: byol
      :link-type: ref
      :class-item: ls-fam-item ls-family-distillation
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Self-distillation · 2020

      An online network predicts the output of a momentum target network, with no negatives.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: SimSiam
      :link: simsiam
      :link-type: ref
      :class-item: ls-fam-item ls-family-distillation
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Self-distillation · 2020

      A Siamese network with a stop-gradient: no negatives, large batches or momentum encoder.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span><span>Tutorial</span></p>

   .. grid-item-card:: DINO
      :link: dino
      :link-type: ref
      :class-item: ls-fam-item ls-family-distillation
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Self-distillation · 2021

      A student network matches the outputs of a momentum-averaged teacher.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: FastSiam
      :link: fastsiam
      :link-type: ref
      :class-item: ls-fam-item ls-family-distillation
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Self-distillation · 2022

      SimSiam that averages several target predictions to train with small batches.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: MSN
      :link: msn
      :link-type: ref
      :class-item: ls-fam-item ls-family-distillation
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Self-distillation · 2022

      Compares the cluster assignments of masked and unmasked views.

      .. raw:: html

         <p class="ls-method-tags"><span>ViT</span></p>

   .. grid-item-card:: PMSN
      :link: pmsn
      :link-type: ref
      :class-item: ls-fam-item ls-family-distillation
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Self-distillation · 2022

      MSN with a custom clustering prior, a power law by default, for imbalanced data.

      .. raw:: html

         <p class="ls-method-tags"><span>ViT</span></p>

   .. grid-item-card:: DINOv2
      :link: dinov2
      :link-type: ref
      :class-item: ls-fam-item ls-family-distillation
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Self-distillation · 2023

      DINO with self-distillation at both the image and the patch level.

      .. raw:: html

         <p class="ls-method-tags"><span>ViT</span></p>

   .. grid-item-card:: MAE
      :link: mae
      :link-type: ref
      :class-item: ls-fam-item ls-family-masked
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Masked image modeling · 2021

      Masks most patches and reconstructs their pixels with a light decoder.

      .. raw:: html

         <p class="ls-method-tags"><span>ViT</span></p>

   .. grid-item-card:: SimMIM
      :link: simmim
      :link-type: ref
      :class-item: ls-fam-item ls-family-masked
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Masked image modeling · 2021

      Like MAE, but the encoder sees both masked and visible patches.

      .. raw:: html

         <p class="ls-method-tags"><span>ViT</span></p>

   .. grid-item-card:: iBOT
      :link: ibot
      :link-type: ref
      :class-item: ls-fam-item ls-family-masked
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Masked image modeling · 2021

      Masked image modeling with an online tokenizer and DINO-style self-distillation.

      .. raw:: html

         <p class="ls-method-tags"><span>ViT</span></p>

   .. grid-item-card:: AIM
      :link: aim
      :link-type: ref
      :class-item: ls-fam-item ls-family-masked
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Masked image modeling · 2024

      Predicts the pixels of image patches autoregressively.

      .. raw:: html

         <p class="ls-method-tags"><span>ViT</span></p>

   .. grid-item-card:: CAPI
      :link: capi
      :link-type: ref
      :class-item: ls-fam-item ls-family-masked
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Masked image modeling · 2025

      Predicts the cluster assignments of masked patches instead of their pixels.

      .. raw:: html

         <p class="ls-method-tags"><span>ViT</span></p>

   .. grid-item-card:: Pixio
      :link: pixio
      :link-type: ref
      :class-item: ls-fam-item ls-family-masked
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Masked image modeling · 2025

      MAE with a much deeper decoder and block-wise masking.

      .. raw:: html

         <p class="ls-method-tags"><span>ViT</span></p>

   .. grid-item-card:: Barlow Twins
      :link: barlowtwins
      :link-type: ref
      :class-item: ls-fam-item ls-family-redundancy
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Redundancy reduction · 2021

      Drives the cross-correlation of two views' embeddings toward the identity.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: VICReg
      :link: vicreg
      :link-type: ref
      :class-item: ls-fam-item ls-family-redundancy
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Redundancy reduction · 2022

      Variance, invariance and covariance terms prevent collapse without negatives.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: VICRegL
      :link: vicregl
      :link-type: ref
      :class-item: ls-fam-item ls-family-redundancy
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Redundancy reduction · 2022

      VICReg on global and local features, for detection and segmentation.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: MMCR
      :link: mmcr
      :link-type: ref
      :class-item: ls-fam-item ls-family-redundancy
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Redundancy reduction · 2023

      Maximizes manifold capacity: the spread of each image's mean view embedding.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: FroSSL
      :link: frossl
      :link-type: ref
      :class-item: ls-fam-item ls-family-redundancy
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Redundancy reduction · 2024

      Avoids collapse by maximizing embedding entropy through Frobenius norms.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: LeJEPA
      :link: lejepa
      :link-type: ref
      :class-item: ls-fam-item ls-family-redundancy
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Redundancy reduction · 2025

      Invariance across views, with embeddings regularized by SIGReg.

      .. raw:: html

         <p class="ls-method-tags"><span>ViT</span></p>

   .. grid-item-card:: SwaV
      :link: swav
      :link-type: ref
      :class-item: ls-fam-item ls-family-clustering
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Clustering and dense · 2020

      Clusters the data and swaps cluster assignments between views.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: DenseCL
      :link: densecl
      :link-type: ref
      :class-item: ls-fam-item ls-family-clustering
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Clustering and dense · 2021

      MoCo with a dense, pixel-level contrastive loss for detection and segmentation.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: DetConS
      :link: detcon
      :link-type: ref
      :class-item: ls-fam-item ls-family-clustering
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Clustering and dense · 2021

      SimCLR on per-region embeddings from unsupervised segmentation masks.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: SMoG
      :link: smog
      :link-type: ref
      :class-item: ls-fam-item ls-family-clustering
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      Clustering and dense · 2022

      Contrastive learning between groups of images instead of single images.

      .. raw:: html

         <p class="ls-method-tags"><span>ResNet</span></p>

   .. grid-item-card:: LeWM
      :link: lewm
      :link-type: ref
      :class-item: ls-fam-item ls-family-world
      :class-card: ls-method-card

      .. rst-class:: ls-method-meta

      World models · 2026

      A latent world model: predicts the next frame's embedding from past frames and actions.

      .. raw:: html

         <p class="ls-method-tags"><span>ViT</span><span>Tutorial</span></p>

.. toctree::
    :hidden:

    aim.rst
    barlowtwins.rst
    byol.rst
    capi.rst
    dcl.rst
    detcon.rst
    densecl.rst
    dino.rst
    dinov2.rst
    fastsiam.rst
    frossl.rst
    ibot.rst
    lejepa.rst
    mae.rst
    mmcr.rst
    msn.rst
    moco.rst
    nnclr.rst
    pixio.rst
    pmsn.rst
    simclr.rst
    simmim.rst
    simsiam.rst
    smog.rst
    swav.rst
    tico.rst
    vicreg.rst
    vicregl.rst
