"""

.. _lightly-tutorial-active-learning-detectron2:

Tutorial 4: Active Learning using Detectron2 on Comma10k
=========================================================

This tutorial is available as a
`Google Colab Notebook <https://colab.research.google.com/drive/1r0KDqIwr6PV3hFhREKgSjRaEbQa5N_5I?usp=sharing>`_

In this tutorial you will learn:

*   how to use Lightly Active Learning together with the
    `detectron2 <https://github.com/facebookresearch/detectron2>`_ framework
    for object detection
*   how to use the Lightly Platform to inspect the selected samples
*   how to get the selected samples for labeling

The tutorial will be divided into
the following steps. 

#. Installation of detectron2 and lightly
#. Run predictions using a pretrained model
#. Use lightly to compute active learning scores for the predictions
#. Use the Lightly Platform to understand where our model struggles
#. Select the most valuable 100 images for annotation

Requirements
------------
- Make sure you have the detectron2 framework installed on your machine. You can use
  the following code to install detectron2:

.. code::
    pip install detectron2

- In this tutorial we work with the comma10k dataset. The dataset consists of
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

# includes for lightly
import lightly
from lightly.active_learning.utils.bounding_box import BoundingBox
from lightly.active_learning.utils.object_detection_output import ObjectDetectionOutput
from lightly.active_learning.scorers import ScorerObjectDetection
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.active_learning.agents import ActiveLearningAgent
from lightly.active_learning.config import SamplerConfig
from lightly.openapi_generated.swagger_client import SamplingMethod

# %%
# Upload dataset to lightly
# -------------------------
#
# To work with the Lightly Platform and use the active learning feature we
# need to uploda the dataset. 
# 
# First, head over to `the Lightly Platform <https://app.lightly.ai/>`_ and 
# create a new dataset.
#
# We can now upload the data using using the command line interface. Don't forget
# to adjust the input_dir to the location of your dataset.
#
# .. code:: 
# 
#     lightly-magic token="yourToken" dataset_id="yourDatasetId" \
#         input_dir='/datasets/comma10k/imgs/' trainer.max_epochs=20 \
#         loader.batch_size=64 loader.num_workers=2
#
# .. note::
#
#     In this tutorial we use the lightly-magic command which trains a model
#     before embedding and uploading it to the Lightly Platform.


YOUR_TOKEN = "yourToken"  # your token of the web platform
YOUR_DATASET_ID = "yourDatasetId"  # the id of your dataset on the web platform
DATASET_ROOT = '/datasets/comma10k/imgs/'

# allow setting of token and dataset_id from environment variables
def try_get_token_and_id_from_env():
    token = os.getenv('TOKEN', YOUR_TOKEN)
    dataset_id = os.getenv('AL_TUTORIAL_DATASET_ID', YOUR_DATASET_ID)
    return token, dataset_id

YOUR_TOKEN, YOUR_DATASET_ID = try_get_token_and_id_from_env()



# %%
# Inference on unlabeled data
# ----------------------------
#
# In active learning we want to pick the new data for which our model struggles 
# the most. If we have an image with a single car in it and our model has a very 
# high confidence that there is a single car we don't gain a lot by including 
# this example into our training data. However, if we focus on images where the 
# model is not sure whether the object is a car or a building we want 
# to include the image.
#
# So what we need to do is provide lightly with the model predictions. 
# Wen can use the ApiWorkflowClient for this. Make sure that we use the 
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

# let's verify the length of the query_set. This should match the number of uploaded
# images (1887)
print(len(al_agent.query_set))

# %%
# Create our Detectron2 model
# ----------------------------
#
#Then, we create a detectron2 config and a detectron2 `DefaultPredictor` to run inference on the newimages.
# 
# - We use a pre-trained Faster R-CNN with a ResNet-50 backbone
# - We use a MS COCO pre-trained model from detectron2

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

def predict_and_overlay(model, fname):
    # helper method to run the model on an image and overlay the predictions
    im = cv2.imread(fname)
    out = model(im)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(out["instances"].to("cpu"))
    plt.figure(figsize=(16,12))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
    plt.tight_layout()

# %%

def convert_bbox_detectron2lightly(outputs):
    # convert detectron2 predictions into lightly format
    height, width = outputs['instances'].image_size
    boxes = []

    for (bbox_raw, score, class_idx) in zip(outputs['instances'].pred_boxes.tensor, outputs['instances'].scores, outputs['instances'].pred_classes):
        x0, y0, x1, y1 = bbox_raw.cpu().numpy()
        x0 /= width
        y0 /= height
        x1 /= width
        y1 /= height
      
        boxes.append(BoundingBox(x0, y0, x1, y1))
    output = ObjectDetectionOutput.from_scores(boxes, outputs['instances'].scores.cpu().numpy(), outputs['instances'].pred_classes.cpu().numpy().tolist())
    return output

# %%
# Run model predictions
# ----------------------
#
# We now use the created model and iterate over the query_set and make predictions.
# It's important that the predictions are in the same order as the filenames
# in the query_set. Otherwise, we could upload a prediction to the wrong sample!

obj_detection_outputs = []
pbar = tqdm.tqdm(al_agent.query_set)
for fname in pbar:
  fname_full = os.path.join(DATASET_ROOT, fname)
  im = cv2.imread(fname_full)
  out = predictor(im)
  obj_detection_output = convert_bbox_detectron2lightly(out)
  obj_detection_outputs.append(obj_detection_output)

# %%

# now, we need to turn the predictions into scores
# The scorer assigns scores between 0.0 and 1.0 to 
# each sample and for each scoring method
scorer = ScorerObjectDetection(obj_detection_outputs)
scores = scorer.calculate_scores()
# %% 
# let's have a look at the sample with the highest
# prediction-margin score
max_score = scores['uncertainty_margin'].max()
idx = scores['uncertainty_margin'].argmax()
print(f'Highest uncertainty_margin score found for idx {idx}: {max_score}')

# %%
# let's have a look at this particular example and show the model 
# prediction
fname = os.path.join(DATASET_ROOT, al_agent.query_set[idx])
predict_and_overlay(predictor, fname)

# %%
config = SamplerConfig(
  n_samples=100, 
  method=SamplingMethod.CORAL, 
  name='active-learning-loop-1'
)
al_agent.query(config, scorer)

# %%
# we can access the newly added data from the agent
print(len(al_agent.added_set))

# %%
# let's have a look at the first 5 entries
print(al_agent.added_set[:5])

# %%
al_agent.query_set.index(al_agent.added_set[0])


# %%
# let's show model predictions for the first 3 images
to_label = [os.path.join(DATASET_ROOT, x) for x in al_agent.added_set]
for i in range(5):
  predict_and_overlay(predictor, to_label[i])