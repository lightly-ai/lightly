"""
.. _lightly-detectron-tutorial-6:

.. role:: bash(code)
   :language: bash

Tutorial 6: Pre-train a Detectron2 Backbone with Lightly
==========================================================

In this tutorial we show how you can do self-supervised pre-training of a
Detectron2 backbone with lightly. The focus of this tutorial is on how to get
and store a pre-trained ResNet50 backbone of the popular Detectron2 framework.
If you want to learn more about self-supervised learning in general, go check
out the following tutorials:

 - :ref:`lightly-moco-tutorial-2`
 - :ref:`lightly-simclr-tutorial-3`:
 - :ref:`lightly-simsiam-tutorial-4`

What you will learn:

- How to retrieve a Detectron2 ResNet50 backbone for pre-training
- How to do self-supervised learning with the backbone
- How to store the backbone in a checkpoint file which can be used by Detectron2

Prerequisites:
----------------
For the purpose of this tutorial you will need to:

- Install Lightly: Follow the `instructions <https://docs.lightly.ai/getting_started/install.html>`_.
- Install Detectron2: Follow the `instructions <https://detectron2.readthedocs.io/en/latest/tutorials/install.html>`_.
- Download a dataset for pre-training (we will use the `SKU-110K <https://github.com/eg4000/SKU110K_CVPR19>`_ dataset).

.. note::

    The SKU-110K dataset is an object detection benchmark for densely packed scenes. It consists of more than 10'000
    images showing retail shelves from varying angles and different camera models. Since contrastive learning works
    especially well on object-centric images, cropping the objects prior to pre-training would be beneficial. However,
    we will use the original images for simplicity. Take a look at `lightly-crop` (:ref:`lightly-command-line-tool`)
    for more info on how to easily extract bounding box crops from your dataset.

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
# We use a batch size of 64 and an input size of 128 in order to fit everything
# on the available amount of memory on our GPU (16GB). The number of features
# is set to the default output size of the ResNet50 backbone.

num_workers = 8
batch_size = 64
input_size = 128
num_ftrs = 2048
lr = 0.001

seed = 1
max_epochs = 0

# use cuda if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# some images in the dataset are truncated, use this to prevent errors
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%
# You might have downloaded the dataset somewhere else or are using a different one.
# Set the path to the dataset accordingly. Additionally, make sure to set the
# path to the config file of the Detectron2 model you want to use.
# We will be using an RCNN with a feature pyramid network (FPN).
data_path = '/datasets/SKU110K_fixed/resized'
cfg_path = './Base-RCNN-FPN.yaml'

# %%
# Initialize the Detectron2 Model
# --------------------------------
# 
# Since the output of the Detectron2 ResNet50 backbone is a dictionary with the keys
# `res1` through `res5`, we have to add an additional layer which picks the right output.
class SelectFeatures(torch.nn.Module):
    """Selects features from a given output layer"""
    
    def __init__(self, features):
        super().__init__()
        self.features = features
    
    def forward(self, x):
        return x[self.features]

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

backbone = torch.nn.Sequential(
    detmodel.backbone.bottom_up,
    SelectFeatures('res5'),
    torch.nn.AdaptiveAvgPool2d(1),
)

# Finally, let's build SimCLR around the backbone as shown in the other
# tutorials.
simclr = lightly.models.simclr.SimCLR(backbone, num_ftrs=num_ftrs)

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
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
    min_scale=0.85,
)

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
optimizer = torch.optim.SGD(simclr.parameters(), lr=lr, momentum=0.9)

gpus = 1 if device == 'cuda' else 0
encoder = lightly.embedding.SelfSupervisedEmbedding(
    simclr,
    criterion,
    optimizer,
    dataloader_train_simclr
)

encoder.train_embedding(gpus=gpus,
                        progress_bar_refresh_rate=100,
                        max_epochs=max_epochs)


# %%
# Storing the checkpoint
# -----------------------
# Now, we can use the pre-trained backbone from the Detectron2 model. The code
# below shows how to save it as a Detectron2 checkpoint called `my_model.pth`.
detmodel.backbone.bottom_up = simclr.backbone[0]

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
#.. note::
#
#   Since the model was pre-trained with images in the RGB input format, it's
#   necessary to set the input format, pixel mean, and pixel std as shown above.


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
