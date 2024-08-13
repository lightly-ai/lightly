"""
.. _lightly-timm-backbone-tutorial-8:

Tutorial 8: Using timm models as backbones
===========================================

You can use LightlySSL to pre-train any model using self-supervised learning since they
share a similar workflow. All methods have a base model (the backbone), which can be any
fundamental model such as ResNet or MobileNet.

In this tutorial, we will learn how to use a model architecture from the timm library
as a backbone in a self-supervised learning workflow.
"""

# %%
# Imports
# -------
#
# Import the Python frameworks we need for this tutorial.
# Make sure you have the necessary packages installed.
#
# .. code-block:: console
#
#   pip install lightly"[timm]"


import timm
import torch
import torch.nn as nn

# %%
# timm comes packaged with >700 pre-trained models designed to be flexible and easy to use.
# These pre-trained models can be loaded using the
# `create_model() <https://huggingface.co/docs/timm/v1.0.8/en/reference/models#timm.create_model>`_
# function. For example, we can use the following snippet to create an efficient model.

model = timm.create_model(model_name="efficientnet_b0")

# %%
# To use a model from timm as a backbone for a self supervised learning method
# we need to remove the last linear layer. We can use the children helper method
# to break down a model into its constituent layers and then convert into a list
# to filter out the necessary layers

backbone = nn.Sequential(
    *list(model.children())[:-1],
)

# %%
# Using a timm model as a backbone
# ---------------------------------
#
# We can now use this model as a backbone for a method, Let's see how we can
# create a torch module for the `SimCLR <https://arxiv.org/abs/2002.05709>`_ method.

from lightly.models.modules.heads import SimCLRProjectionHead


class SimCLR(torch.nn.Module):
    def __init__(self, backbone, feature_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(feature_dim, feature_dim, out_dim)

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z


simclr = SimCLR(backbone, feature_dim=1280, out_dim=128)

# check if it works
input_a = torch.randn((2, 3, 224, 224))
input_b = torch.randn((2, 3, 224, 224))
out_a = simclr(input_a)
out_b = simclr(input_b)

# %%
# Next Steps
# ------------
#
# Interested in pre-training your own self-supervised models? Check out our other
# tutorials:
#
# - :ref:`input-structure-label`
# - :ref:`lightly-moco-tutorial-2`
# - :ref:`lightly-simsiam-tutorial-4`
# - :ref:`lightly-custom-augmentation-5`
# - :ref:`lightly-detectron-tutorial-6`
#
