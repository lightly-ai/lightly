import tempfile
import os
import time

import numpy as np

from lightly.active_learning.agents.agent import ActiveLearningAgent
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.api.bitmask import BitMask
from lightly.data import LightlyDataset
from lightly.api.upload import upload_dataset

from lightly.utils import save_embeddings

from lightly.active_learning.config.sampler_config import SamplerConfig


def t_est_unmocked_complete_workflow(path_to_dataset: str, token: str, dataset_id: str):

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
        filenames = [os.path.basename(filepath) for filepath in filepaths]
        print("Starting save of embeddings")
        save_embeddings(path_to_embeddings_csv, embeddings, labels, filenames)
        print("Finished save of embeddings")

    # upload the embeddings
    print("Starting upload of embeddings")
    api_workflow_client.upload_embeddings(path_to_embeddings_csv=path_to_embeddings_csv, name=f"embedding_1")
    print("Finished upload of embeddings")

    # perform_a_sampling
    print("Starting performing a sampling")
    agent = ActiveLearningAgent(api_workflow_client)
    sampler_config = SamplerConfig(batch_size=8)
    chosen_samples_ids, chosen_filenames = agent.sample(sampler_config=sampler_config)
    print(f'chosen_filenames: {chosen_filenames}')
    print("Finished the sampling")


if __name__ == "__main__":
    path_to_dataset = "/Users/malteebnerlightly/Documents/datasets/clothing-dataset-small-master/test/dress"
    token = os.getenv("TOKEN")
    dataset_id = "602bc1e54dd1e9003219d483"
    for i in range(2):
        print(f"ITERATION {i}:")
        t_est_unmocked_complete_workflow(path_to_dataset, token, dataset_id)
        print("")


