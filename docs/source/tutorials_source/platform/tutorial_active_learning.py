"""

.. _lightly-tutorial-active-learning-knn:

Tutorial 3: Active-Learning with kNN
==============================================

In this tutorial, we will run an active-learning loop using both the lightly package and the platform.
An active-learning loop is a sequence of multiple samplings each choosing only a subset
of all samples in the dataset.

To learn more about how active-learning with lightly works have a look at :ref:`lightly-active-learning`.

This workflow has the following structure:

1. Choose an initial subset of your dataset e.g. using one of our samplers like the coreset sampler.
Split your dataset accordingly into a labeled set and unlabeled set. 

Next, the active-learning loop starts:

2. Train a classifier on the labeled set.

3. Use the classifier to predict on the unlabeled set.

4. Calculate active-learning scores from the prediction.

5. Use an active-learning agent to choose the next samples to be labeled based on the scores.

6. Update the labeled set to include the newly chosen samples and remove them from the unlabeled set.

In this tutorial, we use a k-nearest-neighbor classifier that predicts the class of a sample
based on the class of the k nearest samples in the labeled set.
We use the euclidean distance between a sample's embeddings as the distance metric.
The advantage of such a classifier compared to CNNs is that it is very fast and easily implemented.


What you will learn
-------------------
* You learn how an active-learning loop is set up and which components are needed for it.
* You learn how to perform active-learning with Lightly.

Requirements
------------
- Make sure you are familiar with the command line tools described at :ref:`lightly-command-line-tool`.
- To make the definition of the dataset for training the classifier easy,
  we recommend using a dataset where the samples are grouped into folder according to their class.
  We use the clothing-dataset-small. You can download it using

.. code::

    git clone https://github.com/alexeygrigorev/clothing-dataset-small.git


Creation of the dataset on the Lightly Platform with embeddings
---------------------------------------------------------------

To perform samplings, we need to perform several steps, ideally with the CLI.
More documentation on each step can be found here: :ref:`lightly-command-line-tool`.

A. Train a model on the dataset, e.g. with

.. code::

    lightly-train input_dir="/datasets/clothing-dataset-small/train trainer.max_epochs=5

B. Create embeddings for the dataset, e.g. with

.. code::
    
    lightly-embed input_dir="/datasets/clothing-dataset-small/train" checkpoint=mycheckpoint.ckpt

Save the path to the embeddings.csv, you will need it later.
for uploading the embeddings and for defining the dataset for the classifier

C. Create a new dataset on the lightly platform as described in :ref:`lightly-platform`.
Save the token and dataset id, you will need them later to upload the images and embeddings
and to run the active-learning samplers.

D. Upload the images to the platform, e.g. with

.. code::
    
    lightly-upload input_dir="/datasets/clothing-dataset-small/train" token="yourToken" dataset_id="yourDatasetId"

"""


# %%
# Active-Learning
# -----------------
#
# Import the Python frameworks we need for this tutorial.

import os
import csv
from typing import List, Dict, Tuple
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from lightly.active_learning.agents.agent import ActiveLearningAgent
from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.classification import ScorerClassification
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import SamplingMethod

# %%
# Define the parameters (make sure to set the path to your embeddings file).
path_to_embeddings_csv = "/datasets/clothing-dataset-small/embeddings.csv"
YOUR_TOKEN = "yourToken"  # your token of the web platform
YOUR_DATASET_ID = "yourDatasetId"  # the id of your dataset on the web platform

# allow setting of token and dataset_id from environment variables
def try_get_token_and_id_from_env():
    token = os.getenv('TOKEN', YOUR_TOKEN)
    dataset_id = os.getenv('AL_TUTORIAL_DATASET_ID', YOUR_DATASET_ID)
    return token, dataset_id

YOUR_TOKEN, YOUR_DATASET_ID = try_get_token_and_id_from_env()

# %%
# Define the dataset for the classifier based on the embeddings.csv
# The kNN-classifier needs the embeddings as features for its classification. Thus we define a class to create
# such a dataset out of the embeddings.csv. It also allows to choose only a subset of all samples
# dependent on the filenames given.

class CSVEmbeddingDataset:
    def __init__(self, path_to_embeddings_csv: str):
        with open(path_to_embeddings_csv, 'r') as f:
            data = csv.reader(f)

            rows = list(data)
            header_row = rows[0]
            rows_without_header = rows[1:]

            index_filenames = header_row.index('filenames')
            filenames = [row[index_filenames] for row in rows_without_header]

            index_labels = header_row.index('labels')
            labels = [row[index_labels] for row in rows_without_header]

            embeddings = rows_without_header
            indexes_to_delete = sorted([index_filenames, index_labels], reverse=True)
            for embedding_row in embeddings:
                for index_to_delete in indexes_to_delete:
                    del embedding_row[index_to_delete]

        # create the dataset as a dictionary mapping from the filename to a tuple of the embedding and the label
        self.dataset: Dict[str, Tuple[np.ndarray, int]] = \
            dict([(filename, (np.array(embedding_row, dtype=float), int(label)))
                  for filename, embedding_row, label in zip(filenames, embeddings, labels)])

    def get_features(self, filenames: List[str]) -> np.ndarray:
        features_array = np.array([self.dataset[filename][0] for filename in filenames])
        return features_array

    def get_labels(self, filenames: List[str]) -> np.ndarray:
        labels = np.array([self.dataset[filename][1] for filename in filenames])
        return labels


# %%
# Upload the embeddings to the lightly web platform
api_workflow_client = ApiWorkflowClient(token=YOUR_TOKEN, dataset_id=YOUR_DATASET_ID)
api_workflow_client.upload_embeddings(name="embedding-1", path_to_embeddings_csv=path_to_embeddings_csv)

# %%
# Define the dataset for the classifer, the classifier and the active-learning agent
dataset = CSVEmbeddingDataset(path_to_embeddings_csv=path_to_embeddings_csv)
classifier = KNeighborsClassifier(n_neighbors=20, weights='distance')
agent = ActiveLearningAgent(api_workflow_client=api_workflow_client)

# %%
# 1. Choose an initial subset of your dataset.
# We want to start with 100 samples and use the CORESET sampler for sampling them.
print("Starting the initial sampling")
sampler_config = SamplerConfig(n_samples=100, method=SamplingMethod.CORESET, name='initial-selection')
agent.query(sampler_config=sampler_config)
print(f"There are {len(agent.labeled_set)} samples in the labeled set.")

# %%
# 2. Train a classifier on the labeled set.
labeled_set_features = dataset.get_features(agent.labeled_set)
labeled_set_labels = dataset.get_labels(agent.labeled_set)
classifier.fit(X=labeled_set_features, y=labeled_set_labels)

# %%
# 3. Use the classifier to predict on the unlabeled set.
unlabeled_set_features = dataset.get_features(agent.unlabeled_set)
predictions = classifier.predict_proba(X=unlabeled_set_features)

# %%
# 4. Calculate active-learning scores from the prediction.
active_learning_scorer = ScorerClassification(model_output=predictions)

# %%
# 5. Use an active-learning agent to choose the next samples to be labeled based on the active-learning scores.
# We want to sample another 100 samples to have 200 samples in total and use the active-learning sampler CORAL for it.
sampler_config = SamplerConfig(n_samples=200, method=SamplingMethod.CORAL, name='al-iteration-1')
agent.query(sampler_config=sampler_config, al_scorer=active_learning_scorer)
print(f"There are {len(agent.labeled_set)} samples in the labeled set.")

# %%
# 6. Update the labeled set to include the newly chosen samples and remove them from the unlabeled set.
# This is already done internally inside the ActiveLearningAgent - no work for you :)

# %%
# Now you can use the newly chosen labeled set to retrain you classifer on it.
# You can evaluate it e.g. on the unlabeled set, or on embeddings of a test set you generated before.
# If you are not satisfied with the performance, you can run steps 2 to 5 again.
labeled_set_features = dataset.get_features(agent.labeled_set)
labeled_set_labels = dataset.get_labels(agent.labeled_set)
classifier.fit(X=labeled_set_features, y=labeled_set_labels)

# evaluate on unlabeled set
unlabeled_set_features = dataset.get_features(agent.unlabeled_set)
unlabeled_set_labels = dataset.get_labels(agent.unlabeled_set)
accuracy_on_unlabeled_set = classifier.score(X=unlabeled_set_features, y=unlabeled_set_labels)
print(f"accuracy on unlabeled set: {accuracy_on_unlabeled_set}")
