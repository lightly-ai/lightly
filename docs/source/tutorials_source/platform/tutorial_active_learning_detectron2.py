"""

.. _lightly-tutorial-active-learning-detectron2:

Tutorial 4: Active Learning using Detectron2 on Comma10k
=========================================================

Active learning is a process of using model predictions to find a new set of
images to annotate. The images are chosen to have a maximal impact on the model
performance. In this tutorial, we will use a pre-trained object detection model
to do active learning on a completely unlabeled set of images.

.. figure:: images/sphx_glr_tutorial_active_learning_detectron2_003.png
   :align: center
   :alt: Detectron2 Faster RCNN prediction on Comma10k

   Detectron2 Faster RCNN prediction on Comma10k

In machine learning, we often don't train a model from scratch.
Instead, we start with an already pre-trained model. For object detection tasks,
a common pre-training dataset is MS COCO consisting of over 100'000 images
containing 80 different classes. Our goal is to take an MS COCO pre-trained
model and optimize it for an autonomous driving task. We will proceed as
follows: First, we will use the pre-trained model to make predictions on our
task dataset (Comma10k) which has been collected for autonomous driving. Then,
we use the predictions, self-supervised learning, and active learning with the
lightly framework to find the 100 most informative images on which we can
finetune our model.

This tutorial is available as a
`Google Colab Notebook <https://colab.research.google.com/drive/1r0KDqIwr6PV3hFhREKgSjRaEbQa5N_5I?usp=sharing>`_

In this tutorial you will learn:

*   how to use Lightly Active Learning together with the
    `detectron2 <https://github.com/facebookresearch/detectron2>`_ framework
    for object detection
*   how to use the Lightly Platform to inspect the selected samples
*   how to download the selected samples for labeling

The tutorial will be divided into
the following steps. 

#. Installation of detectron2 and lightly
#. Run predictions using a pre-trained model
#. Use lightly to compute active learning scores for the predictions
#. Use the Lightly Platform to understand where our model struggles
#. Select the most valuable 100 images for annotation

Requirements
------------
- Make sure you have OpenCV installed to read and preprocess the images.
  You can install the framework using the following command:

.. code::
  
   pip install opencv-python


- Make sure you have the detectron2 framework installed on your machine. Check out
  the `detectron2 installation documentation <https://detectron2.readthedocs.io/en/latest/tutorials/install.html>`_

- In this tutorial, we work with the comma10k dataset. The dataset consists of
  10'000 images for autonomous driving and is available
  `here on GitHub <https://github.com/commaai/comma10k>`_
  We can download the dataset using `git clone`. We save the dataset locally
  to `/datasets/`

.. code::

    git clone https://github.com/commaai/comma10k

"""

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, glob
import tqdm, gc
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# imports for lightly
import lightly
from lightly.active_learning.utils.bounding_box import BoundingBox
from lightly.active_learning.utils.object_detection_output import ObjectDetectionOutput
from lightly.active_learning.scorers import ScorerObjectDetection
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.active_learning.agents import ActiveLearningAgent
from lightly.active_learning.config import SelectionConfig
from lightly.openapi_generated.swagger_client import SamplingMethod

# %%
# Upload dataset to Lightly
# -------------------------
#
# To work with the Lightly Platform and use the active learning feature we
# need to upload the dataset. 
# 
# First, head over to `the Lightly Platform <https://app.lightly.ai/>`_ and 
# create a new dataset.
#
# We can now upload the data using the command line interface. Replace 
# **yourToken** and **yourDatasetId** with the two provided values from the web app.
# Don't forget to adjust the **input_dir** to the location of your dataset.
#
# .. code:: 
# 
#     lightly-magic token="yourToken" dataset_id="yourDatasetId" \
#         input_dir='/datasets/comma10k/imgs/' trainer.max_epochs=20 \
#         loader.batch_size=64 loader.num_workers=8
#
# .. note::
#
#     In this tutorial, we use the lightly-magic command which trains a model
#     before embedding and uploading it to the Lightly Platform.
#     To skip training, you can set `trainer.max_epochs=0`.


YOUR_TOKEN = "yourToken"  # your token of the web platform
YOUR_DATASET_ID = "yourDatasetId"  # the id of your dataset on the web platform
DATASET_ROOT = '/datasets/comma10k/imgs/'

# allow setting of token and dataset_id from environment variables
def try_get_token_and_id_from_env():
    token = os.getenv('TOKEN', YOUR_TOKEN)
    print(token[:3])
    dataset_id = os.getenv('AL_TUTORIAL_DATASET_ID', YOUR_DATASET_ID)
    return token, dataset_id

YOUR_TOKEN, YOUR_DATASET_ID = try_get_token_and_id_from_env()



# %%
# Inference on unlabeled data
# ----------------------------
#
# In active learning, we want to pick the new data for which our model struggles 
# the most. If we have an image with a single car in it and our model has 
# high confidence that there is a car we don't gain a lot by including 
# this example in our training data. However, if we focus on images where the 
# model is not sure whether the object is a car or a building we want 
# to include these images to refine the decision boundary.
#
# First, we need to create an active learning agent in order to 
# provide lightly with the model predictions. 
# We can use the ApiWorkflowClient for this. Make sure that we use the 
# right dataset_id and token.

# create Lightly API client
api_client = ApiWorkflowClient(dataset_id=YOUR_DATASET_ID, token=YOUR_TOKEN)
al_agent = ActiveLearningAgent(api_client)

# %%

# we can access the images of the dataset we want to use for active learning using
# the `al_agent.query_set` property

# let's print the first 3 entries
print(al_agent.query_set[:3])

# %%
# Note, that our active learning agent already synchronized with the Lightly
# Platform and knows the filenames present in our dataset.
#
# Let's verify the length of the `query_set`. The `query_set` is the set of 
# images from which we want to query. By default this is our full
# dataset uploaded to Lightly. You can learn more about the different sets we 
# can access through the active learning agent here
# :py:class:`lightly.api.api_workflow_client.ApiWorkflowClient`


# The length of the `query_set` should match the number of uploaded
# images
print(len(al_agent.query_set))

# %%
# Create our Detectron2 model
# ----------------------------
#
# Next, we create a detectron2 config and a detectron2 `DefaultPredictor` to 
# run predictions on the new images.
# 
# - We use a pre-trained Faster R-CNN with a ResNet-50 backbone
# - We use an MS COCO pre-trained model from detectron2

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
###cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
###cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# %%
# We use this little helper method to overlay the model predictions on a
# given image.
def predict_and_overlay(model, filename):
    # helper method to run the model on an image and overlay the predictions
    im = cv2.imread(filename)
    out = model(im)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(out["instances"].to("cpu"))
    plt.figure(figsize=(16,12))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
    plt.tight_layout()

# %%
# The lightly framework expects a certain bounding box and prediction format. 
# We create another helper method to convert the detectron2 output into 
# the desired format.
def convert_bbox_detectron2lightly(outputs):
    # convert detectron2 predictions into lightly format
    height, width = outputs['instances'].image_size
    boxes = []

    for (bbox_raw, score, class_idx) in zip(outputs['instances'].pred_boxes.tensor, 
                                            outputs['instances'].scores,
                                            outputs['instances'].pred_classes):
        x0, y0, x1, y1 = bbox_raw.cpu().numpy()
        x0 /= width
        y0 /= height
        x1 /= width
        y1 /= height
      
        boxes.append(BoundingBox(x0, y0, x1, y1))
    output = ObjectDetectionOutput.from_scores(
      boxes, outputs['instances'].scores.cpu().numpy(),
      outputs['instances'].pred_classes.cpu().numpy().tolist())
    return output

# %%
# Get Model Predictions
# ----------------------
#
# We now use the created model and iterate over the `query_set` and make predictions.
# It's important that the predictions are in the same order as the filenames
# in the `query_set`. Otherwise, we could upload a prediction to the wrong sample!

obj_detection_outputs = []
pbar = tqdm.tqdm(al_agent.query_set, miniters=500, mininterval=60, maxinterval=120)
for fname in pbar:
  fname_full = os.path.join(DATASET_ROOT, fname)
  im = cv2.imread(fname_full)
  out = predictor(im)
  obj_detection_output = convert_bbox_detectron2lightly(out)
  obj_detection_outputs.append(obj_detection_output)

# %%
# Now, we need to turn the predictions into scores.
# The scorer assigns scores between 0.0 and 1.0 to 
# each sample and for each scoring method.
scorer = ScorerObjectDetection(obj_detection_outputs)
scores = scorer.calculate_scores()
# %% 
# Let's have a look at the sample with the highest
# uncertainty_margin score.
#
# .. note::
#    A high uncertainty margin means that the image contains at least one 
#    bounding box for which the model is unsure about the class of the object 
#    in the bounding box. Read more about how our active learning scores are 
#    calculated here:
#    :py:class:`lightly.active_learning.scorers.detection.ScorerObjectDetection`
max_score = scores['uncertainty_margin'].max()
idx = scores['uncertainty_margin'].argmax()
print(f'Highest uncertainty_margin score found for idx {idx}: {max_score}')

# %%
# Let's have a look at this particular image and show the model 
# prediction for it.
fname = os.path.join(DATASET_ROOT, al_agent.query_set[idx])
predict_and_overlay(predictor, fname)

# %%
# Query Samples for Labeling
# ---------------------------
#
# Finally, we can tell our agent to select the top 100 images to annotate and
# improve our existing model. We pick the selection strategy called `CORAL` which
# is a combination of CORESET and Active Learning. Whereas CORESET maximizes
# the image diversity based on the embeddings, active learning aims at selecting
# images where our model struggles the most.
config = SelectionConfig(
  n_samples=100, 
  method=SamplingMethod.CORAL, 
  name='active-learning-loop-1'
)
al_agent.query(config, scorer)

# %%
# We can access the newly added data from the agent.
print(len(al_agent.added_set))

# %%
# Let's have a look at the first 5 entries.
print(al_agent.added_set[:5])

# %%
# Let's show model predictions for the first 5 images.
to_label = [os.path.join(DATASET_ROOT, x) for x in al_agent.added_set]
for i in range(5):
  predict_and_overlay(predictor, to_label[i])

# %%
# Samples selected in the step above were placed in the 'active-learning-loop-1' tag.
# This can be viewed on the `Lightly Platform <https://app.lightly.ai/datasets>`_.

# %%
# To re-use a dataset without tags from past experiments, we can (optionally!) remove 
# tags other than the initial-tag:

for tag in api_client.get_all_tags():
  if tag.prev_tag_id is not None:
    api_client.delete_tag_by_id(tag.id)

# %%
# Next Steps
# -------------
# 
# We showed in this tutorial how you can use Lightly Active Learning to discover 
# the images you should label next. You can close the loop by annotating 
# the 100 images and re-training your model. Then start the next iteration 
# by making new model predictions on the `query_set`.
#
# Using Lightly Active Learning has two advantages:
#
# - By letting the model chose the next batch of images to label we achieve 
#   a higher accuracy faster. We're only labeling the images having a great impact.
# 
# - By combining the model predictions with the image embeddings we make sure we 
#   don't select many similar images. Imagine the model being very bad at small 
#   red cars and the 100 images therefore would only contain this set of images. 
#   We might overfit the model because it suddenly has too many training examples 
#   of small red cars.

# %%
# After re-training our model on the newly labeled 100 images 
# we can do another active learning iteration by running predictions on the
# the `query_set`.
