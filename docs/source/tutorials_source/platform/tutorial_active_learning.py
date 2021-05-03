"""

.. _lightly-tutorial-active-learning-classification:

Tutorial 3: Active learning for classification
==============================================

We provide the tutorial in a ready to use 
`Google Colab <https://colab.research.google.com/drive/1E3rz7fY7UqXNI_VYNxSu6KvQINzotwrz?usp=sharing>`_ 
notebook:


In this tutorial, we will run an active learning loop using both the lightly package and the Lightly platform.
An active learning loop is a sequence of multiple samplings each choosing only a subset
of all samples in the dataset.

To learn more about how active learning with lightly works have a look at :ref:`lightly-active-learning`.

This workflow has the following structure:

1. Choose an initial subset of your dataset, e.g. using one of our samplers like the coreset sampler.
Label this initial subset and train your model on it.

Next, the active learning loop starts:

2. Train a classifier on the labeled set.

3. Use the classifier to predict on the unlabeled set.

4. Calculate active learning scores from the prediction.

5. Use an active learning agent to choose the next samples to be labeled based on the scores.

6. Update the labeled set to include the newly chosen samples and remove them from the unlabeled set.

In this tutorial, we use logistic regression on top of the embeddings as a classifier.
This is the same as using the embedding model as a pretrained backbone
and putting a single layer classification head on top of it while fine-tuning only the classification head
on the labeled dataset, but keeping the backbone frozen.
Since the embeddings are already computed, we can use them directly as input to the classification head.
This saves computational time and makes the tutorial quick to walk through.

What you will learn
-------------------
* You learn how an active learning loop is set up and which components are needed for it.
* You learn how to do active learning with Lightly.

Define your dataset
------------
- To make the definition of the dataset for training the classifier easy,
  we recommend using a dataset where the images are grouped in folders by class.
  We use the clothing-dataset-small. You can download it using

.. code::

    git clone https://github.com/alexeygrigorev/clothing-dataset-small.git

The dataset's images are RGB images with a few hundred pixels in width and height. They show clothes
from 10 different classes, like dresses, hats or t-shirts. The dataset is already split into a train,
test, and validation set, and all images for one class are put into one folder.

.. image:: ./platform/images/clothing-dataset-small-structure.png
    :align: center
    :alt: The directory and file structure of the clothing dataset small


Creation of the dataset on the Lightly Platform with embeddings
---------------------------------------------------------------

To do active learning, we need a dataset with embeddings on the Lightly platform.
The first step for this is to train a self-supervised embedding model. Then, embed your dataset and lastly,
upload the dataset and embeddings to the Lightly platform.
These three steps can be done using a single command from the lightly pip package: lightly-magic

Following Commands are all in terminal/command line

.. code::

    # Install lightly as a pip package
    pip install lightly

.. code::

    # Your personal token for accessing the Lightly Platform is defined. You can get it from
    # the Lightly platform at https://app.lightly.ai under your username and then Preferences.
    export LIGHTLY_TOKEN="YOUR_TOKEN"

.. code::

    # The lightly-magic command first needs the input directory of your dataset.
    # Then it needs the information for how many epochs to train an embedding model on it.
    # If you want to use our pretrained model instead, set trainer.max_epochs=0.
    # Next, the embedding model is used to embed all images in the input directory and saves the embeddings in
    # a csv file. Last, a new dataset with the specified name is created on the Lightly platform.
    # The embeddings file is uploaded to it and the images themselves are uploaded with 8 workers in parallel.
    lightly-magic input_dir="./clothing-dataset-small/train" trainer.max_epochs=0 token=$LIGHTLY_TOKEN new_dataset_name="active_learning_clothing_dataset" loader.num_workers=8

.. code::

    # In the console output of the lightly-magic command, you find the filename of the created
    # embeddings file. We need this file later, so set the path to it as an environment variable.
    export LIGHTLY_EMBEDDINGS_CSV="path_to_the_embeddings_csv"

.. code::

    # Install the pip packages required for the tutorial if not already installed
    pip install numpy
    pip install scikit-learn


Optional:
You can find out more about the CLI commands and their parameters at https://docs.lightly.ai/lightly.cli.html.

Optional:
You can have a look at your dataset and embeddings by browsing through it
on the Lightly platform at https://app.lightly.ai.

"""


# %%
# Active learning
# -----------------
#
# Import the Python frameworks we need for this tutorial.

import os
import csv
from typing import List, Dict, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression

from lightly.active_learning.agents.agent import ActiveLearningAgent
from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.classification import ScorerClassification
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import SamplingMethod

# %%
# Define the dataset for the classifier based on the embeddings.csv
# The LogisticRegression classifier needs the embeddings as features for its classification.
# Thus we define a class to create such a dataset out of the embeddings.csv.
# It also allows to choose only a subset of all samples dependant on the filenames given.


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
# First we read the variables we set before as environment variables via the console
token = os.getenv("LIGHTLY_TOKEN", default="YOUR_TOKEN")
path_to_embeddings_csv = os.getenv("LIGHTLY_EMBEDDINGS_CSV", default="path_to_your_embeddings_csv")

# We define the client to the Lightly Platform API
api_workflow_client = ApiWorkflowClient(token=token)
api_workflow_client.create_dataset(dataset_name="active_learning_clothing_dataset")

# %%
# We define the dataset, the classifier and the active learning agent
dataset = CSVEmbeddingDataset(path_to_embeddings_csv=path_to_embeddings_csv)
classifier = LogisticRegression(max_iter=1000)
agent = ActiveLearningAgent(api_workflow_client=api_workflow_client)

# %%
# 1. Choose an initial subset of your dataset.
# We want to start with 200 samples and use the CORESET sampler for sampling them.
print("Starting the initial sampling")
sampler_config = SamplerConfig(n_samples=200, method=SamplingMethod.CORESET, name='initial-selection')
agent.query(sampler_config=sampler_config)
print(f"There are {len(agent.labeled_set)} samples in the labeled set.")

# %%
# 2. Train a classifier on the labeled set.
labeled_set_features = dataset.get_features(agent.labeled_set)
labeled_set_labels = dataset.get_labels(agent.labeled_set)
classifier.fit(X=labeled_set_features, y=labeled_set_labels)

# %%
# 3. Use the classifier to predict on the query set.
query_set_features = dataset.get_features(agent.query_set)
predictions = classifier.predict_proba(X=query_set_features)

# %%
# 4. Calculate active learning scores from the prediction.
active_learning_scorer = ScorerClassification(model_output=predictions)

# %%
# 5. Use an active learning agent to choose the next samples to be labeled based on the active learning scores.
# We want to sample another 100 samples to have 300 samples in total and use the active learning sampler CORAL for it.
sampler_config = SamplerConfig(n_samples=300, method=SamplingMethod.CORAL, name='al-iteration-1')
agent.query(sampler_config=sampler_config, al_scorer=active_learning_scorer)
print(f"There are {len(agent.labeled_set)} samples in the labeled set.")

# %%
# 6. Update the labeled set to include the newly chosen samples and remove them from the unlabeled set.
# This is already done internally inside the ActiveLearningAgent - no work for you :)

# %%
# Now you can use the newly chosen labeled set to retrain your classifier on it.
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
