"""
.. _lightly-detectron-tutorial-6:

.. role:: bash(code)
   :language: bash

Tutorial 6: Pre-train a Detectron2 Backbone with Lightly
==========================================================

In this tutorial we show how you can do self-supervised pre-training of a
`Detectron2 <https://github.com/facebookresearch/detectron2>`_ backbone with lightly.
The focus of this tutorial is on how to get and store a pre-trained ResNet50
backbone of the popular Detectron2 framework. If you want to learn more about
self-supervised learning in general, go check out the following tutorials:

 - :ref:`lightly-moco-tutorial-2`
 - :ref:`lightly-simclr-tutorial-3`:
 - :ref:`lightly-simsiam-tutorial-4`

What you will learn:

- How to retrieve a Detectron2 ResNet50 backbone for pre-training
- How to do self-supervised learning with the backbone
- How to store the backbone in a checkpoint file which can be used by Detectron2

Introduction
----------------

For many tasks in computer vision it can be beneficial to pre-train a neural network
on a domain-specific dataset prior to finetuning it. For example, a retail detection
network can be pre-trained with self-supervised learning on a large retail detection
dataset. This way the neural network learns to extract relevant features from the images
without requiring any annotations at all. As a consequence, it's possible to finetune
the network with only a handful of annotated images. This tutorial will guide you
through the steps to pre-train a detection backbone from the popular
`Detectron2 <https://github.com/facebookresearch/detectron2>`_ framework.

Prerequisites:
----------------
For the purpose of this tutorial you will need to:

- Install Lightly: Follow the `instructions <https://docs.lightly.ai/getting_started/install.html>`_.
- Install Detectron2: Follow the `instructions <https://detectron2.readthedocs.io/en/latest/tutorials/install.html>`_.
- Download a dataset for pre-training (we will use the `Freiburg Groceries Dataset <https://github.com/PhilJd/freiburg_groceries_dataset>`_ dataset). You can download it by cloning the Github repository and running `download_dataset.py`. Alternatively, you can use this `download link <http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz>`_

.. note::

    The `Freiburg Groceries Dataset <https://github.com/PhilJd/freiburg_groceries_dataset>`_ consists of 5000 256x256 RGB images of 25 food classes.
    Images show one or multiple instances of grocery products in shelves or similar scenarios.

Finally, you will need the Detectron2 configuration files. They are available `here <https://github.com/facebookresearch/detectron2/tree/main/configs>`_.
In this tutorial we use a Faster RCNN with a feature pyramid network (FPN), so make sure you have the relevant file (Base-RCNN-FPN.yaml) in your directory.


"""

# %%
# Imports
# -------
#
# Import the Python frameworks we need for this tutorial.
import torch
import lightly
from detectron2 import config, modeling
from detectron2.checkpoint import DetectionCheckpointer

# %%
# Configuration
# -------------
# Let's set the configuration parameters for our experiments.
#
# We use a batch size of 512 and an input size of 128 in order to fit everything
# on the available amount of memory on our GPU (16GB). The number of features
# is set to the default output size of the ResNet50 backbone.
#
# We only train for 5 epochs because the focus of this tutorial is on the
# integration with Detectron2.

num_workers = 8
batch_size = 512
input_size = 128
num_ftrs = 2048

seed = 1
max_epochs = 0

# use cuda if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# %%
# You might have downloaded the dataset somewhere else or are using a different one.
# Set the path to the dataset accordingly. Additionally, make sure to set the
# path to the config file of the Detectron2 model you want to use.
# We will be using an RCNN with a feature pyramid network (FPN).
data_path = '/datasets/freiburg_groceries_dataset/images'
cfg_path = './Base-RCNN-FPN.yaml'

# %%
# Initialize the Detectron2 Model
# --------------------------------
# 
# The output of the Detectron2 ResNet50 backbone is a dictionary with the keys
# `res1` through `res5` (see the `documentation <https://detectron2.readthedocs.io/en/latest/modules/modeling.html#detectron2.modeling.ResNet>`_).
# The keys correspond to the different stages of the ResNet. In this tutorial, we are only
# interested in the high-level abstractions from the last layer, `res5`. Therefore,
# we have to add an additional layer which picks the right output from the dictionary.
class SelectStage(torch.nn.Module):
    """Selects features from a given stage."""

    def __init__(self, stage: str = 'res5'):
        super().__init__()
        self.stage = stage

    def forward(self, x):
        return x[self.stage]

# %%
# Let's load the config file and make some adjustments to ensure smooth training.
cfg = config.get_cfg()
cfg.merge_from_file(cfg_path)

# use cuda if possible
cfg.MODEL.DEVICE = device

# randomly initialize network
cfg.MODEL.WEIGHTS = ""

# detectron2 uses BGR by default but pytorch/torchvision use RGB
cfg.INPUT.FORMAT = "RGB"

# %%
# Next, we can build the Detectron2 model and extract the ResNet50 backbone as
# follows:

detmodel = modeling.build_model(cfg)

simclr_backbone = torch.nn.Sequential(
    detmodel.backbone.bottom_up,
    SelectStage('res5'),
    # res5 has shape bsz x 2048 x 4 x 4
    torch.nn.AdaptiveAvgPool2d(1),
).to(device)

# %%
#
#
#.. note::
#
#   The Detectron2 ResNet is missing the average pooling layer used to get a tensor of shape bsz x 2048.
#   Therefore, we add an average pooling as in the `PyTorch ResNet <https://github.com/pytorch/pytorch/blob/1022443168b5fad55bbd03d087abf574c9d2e9df/benchmarks/functional_autograd_benchmark/torchvision_models.py#L147>`_.
#

# %%
# Finally, let's build SimCLR around the backbone as shown in the other
# tutorials. For this, we only require an additional projection head.
projection_head = lightly.models.modules.SimCLRProjectionHead(
    input_dim=num_ftrs,
    hidden_dim=num_ftrs,
    output_dim=128,
).to(device)

# %%
# Setup data augmentations and loaders
# ------------------------------------
#
# We start by defining the augmentations which should be used for training.
# We use the same ones as in the SimCLR paper but change the input size and
# minimum scale of the random crop to adjust to our dataset. 
#
# We don't go into detail here about using the optimal augmentations.
# You can learn more about the different augmentations and learned invariances
# here: :ref:`lightly-advanced`.
collate_fn = lightly.data.SimCLRCollateFunction(input_size=input_size)

dataset_train_simclr = lightly.data.LightlyDataset(input_dir=data_path)

dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)

# %%
# Self-supervised pre-training
# -----------------------------
# Now all we need to do is define a loss and optimizer and start training!

criterion = lightly.loss.NTXentLoss()
optimizer = torch.optim.Adam(
    list(simclr_backbone.parameters()) + list(projection_head.parameters()),
    lr=1e-4,
)


for e in range(max_epochs):

    mean_loss = 0.
    for (x0, x1), _, _ in dataloader_train_simclr:

        x0 = x0.to(device)
        x1 = x1.to(device)

        y0 = projection_head(simclr_backbone(x0).flatten(start_dim=1))
        y1 = projection_head(simclr_backbone(x1).flatten(start_dim=1))

        # backpropagation
        loss = criterion(y0, y1)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # update average loss
        mean_loss += loss.detach().cpu().item() / len(dataloader_train_simclr)

    print(f'[Epoch {e:2d}] Mean Loss = {mean_loss:.2f}')


# %%
# Storing the checkpoint
# -----------------------
# Now, we can use the pre-trained backbone from the Detectron2 model. The code
# below shows how to save it as a Detectron2 checkpoint called `my_model.pth`.

# get the first module from the backbone (i.e. the detectron2 ResNet)
# backbone:
#     L ResNet50
#     L SelectStage
#     L AdaptiveAvgPool2d
detmodel.backbone.bottom_up = simclr_backbone[0]

checkpointer = DetectionCheckpointer(detmodel, save_dir='./')
checkpointer.save('my_model')


# %%
# Finetuning with Detectron2
# ---------------------------
#
# The checkpoint from above can now be used by any Detectron2 script. For example,
# you can use the `train_net.py` script in the Detectron2 `tools`:
#
#

# %%
#.. code-block:: none
#
#   python train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
#       MODEL.WEIGHTS path/to/my_model.pth \
#       MODEL.PIXEL_MEAN 123.675,116.280,103.530 \
#       MODEL.PIXEL_STD 58.395,57.120,57.375 \
#       INPUT.FORMAT RGB
#

# %%
# 
# The :py:class:`lightly.data.collate.SimCLRCollateFunction` applies an ImageNet
# normalization of the input images by default. Therefore, we have to normalize
# the input images at training time, too. Since Detectron2 uses an input space
# in the range 0 - 255, we use the numbers above.
# 

# %%
#
#.. note::
#
#   Since the model was pre-trained with images in the RGB input format, it's
#   necessary to set the permute the order of the pixel mean, and pixel std as shown above.

# %%
# Next Steps
# ------------
#
#
# Want to learn more about our self-supervised models and how to choose
# augmentations properly? Check out our other tutorials:
#
# - :ref:`lightly-moco-tutorial-2`
# - :ref:`lightly-simclr-tutorial-3`
# - :ref:`lightly-simsiam-tutorial-4`
# - :ref:`lightly-custom-augmentation-5`
