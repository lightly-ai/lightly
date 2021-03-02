import sys
import tempfile
import os
import time

import numpy as np

from lightly.active_learning.agents.agent import ActiveLearningAgent
from lightly.active_learning.scorers.classification import ScorerClassification
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.api.bitmask import BitMask
from lightly.data import LightlyDataset
from lightly.api.upload import upload_dataset
from lightly.openapi_generated.swagger_client import SamplingMethod

from lightly.utils import save_embeddings

from lightly.active_learning.config.sampler_config import SamplerConfig


def t_est_unmocked_complete_workflow(path_to_dataset: str, token: str, dataset_id: str,
                                     preselected_tag_name: str = None, query_tag_name: str = None,
                                     with_scores: bool = True):
    # define the api_client and api_workflow
    api_workflow_client = ApiWorkflowClient(host="https://api-dev.lightly.ai", token=token, dataset_id=dataset_id)

    # upload the images to the dataset and create the initial tag
    no_tags_on_server = len(api_workflow_client.tags_api.get_tags_by_dataset_id(dataset_id=dataset_id))
    if no_tags_on_server == 0:
        api_workflow_client.upload_dataset(input=path_to_dataset)
    else:
        print("Skip upload of dataset: already uploaded.")

    # calculate and save the embeddings
    path_to_embeddings_csv = f"{path_to_dataset}/embeddings.csv"
    if not os.path.isfile(path_to_embeddings_csv):
        dataset = LightlyDataset(input_dir=path_to_dataset)
        embeddings = np.random.normal(size=(len(dataset.dataset.samples), 32))
        filepaths, labels = zip(*dataset.dataset.samples)
        filenames = [filepath[len(path_to_dataset):].lstrip('/') for filepath in filepaths]
        print("Starting save of embeddings")
        save_embeddings(path_to_embeddings_csv, embeddings, labels, filenames)
        print("Finished save of embeddings")

    # upload the embeddings
    print("Starting upload of embeddings")
    api_workflow_client.upload_embeddings(path_to_embeddings_csv=path_to_embeddings_csv, name=f"embedding_1")
    print("Finished upload of embeddings")

    # perform_a_sampling
    print("Starting the AL loop")

    agent = ActiveLearningAgent(api_workflow_client,
                                query_tag_name=query_tag_name, preselected_tag_name=preselected_tag_name)
    total_initial_chosen_samples = len(agent.labeled_set)
    total_no_samples = len(agent.unlabeled_set) + len(agent.labeled_set)
    for iter, n_samples in enumerate([2, 5, 9]):
        total_samples_to_choose = total_initial_chosen_samples + n_samples
        if iter == 0 or not with_scores:
            sampler_config = SamplerConfig(n_samples=total_samples_to_choose)
            chosen_filenames = agent.query(sampler_config=sampler_config)
        else:
            predictions = np.random.rand(len(agent.unlabeled_set), 10)
            predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
            al_scorer = ScorerClassification(predictions_normalized)
            sampler_config = SamplerConfig(n_samples=total_samples_to_choose, method=SamplingMethod.CORAL)
            chosen_filenames = agent.query(sampler_config=sampler_config, al_scorer=al_scorer)
        assert (len(chosen_filenames) == total_samples_to_choose)
        assert (len(agent.labeled_set) == total_samples_to_choose)
        assert (len(agent.unlabeled_set) == total_no_samples - total_samples_to_choose)
        print(f"Finished AL step with {len(chosen_filenames)} labeled samples in total")

    print("Finished the AL loop")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        path_to_dataset = "/Users/malteebnerlightly/Documents/datasets/clothing-dataset-small-master/test"
        token = os.getenv("TOKEN")
        dataset_id = "603e5426e25693003383fcc9"
        query_tag_name = "initial-tag"
        preselected_tag_name = None#"preselected_8_images"
        with_scores = "True"
    elif len(sys.argv) == 1 + 6:
        path_to_dataset, token, dataset_id, query_tag_name, preselected_tag_name, with_scores = \
            (sys.argv[1 + i] for i in range(6))
    else:
        raise ValueError("ERROR in number of command line arguments, must be 6."
                         "Example: python test_al.py this/is/my/dataset "
                         "TOKEN dataset_id query_tag_name preselected_tag_name True")

    if query_tag_name == "None":
        query_tag_name = None
    if preselected_tag_name == "None":
        preselected_tag_name = None
    if with_scores == "True":
        with_scores = True
    elif with_scores == "False":
        with_scores = False
    else:
        raise ValueError("'with_scores' must be either 'True' or 'False'")

    for i in range(1):
        print(f"ITERATION {i}:")
        t_est_unmocked_complete_workflow(path_to_dataset, token, dataset_id,
                                         query_tag_name=query_tag_name, preselected_tag_name=preselected_tag_name,
                                         with_scores=with_scores)
        print("")
