import tempfile
import os
import time

import numpy as np

from lightly.api.api_workflow import ApiWorkflow

from lightly.data import LightlyDataset
from lightly.api.upload import upload_dataset

from lightly.core import embed_images
from lightly.openapi_generated.swagger_client import InitialTagCreateRequest
from lightly.utils import save_embeddings

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.api.bitmask import BitMask


def t_est_unmocked_complete_workflow(path_to_dataset: str, token: str, dataset_id: str):

    # define the api_client and api_workflow
    api_workflow = ApiWorkflow(host="https://api-dev.lightly.ai", token=token, dataset_id=dataset_id)

    # upload the images to the dataset and create the initial tag
    no_created_tags = len(api_workflow.tags_api.get_tags_by_dataset_id(dataset_id=dataset_id))
    if no_created_tags == 0:
        dataset = LightlyDataset(input_dir=path_to_dataset)
        print("Starting upload of dataset")
        upload_dataset(dataset=dataset, dataset_id=dataset_id, token=token, max_workers=12)
        print("Finished creation of intial tag")
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
    api_workflow.upload_embeddings(path_to_embeddings_csv=path_to_embeddings_csv, name=f"embedding_1")
    print("Finished upload of embeddings")

    # perform_a_sampling
    print("Starting performing a sampling")
    sampler_config = SamplerConfig(batch_size=8)
    new_tag = api_workflow.sampling(sampler_config=sampler_config)
    print("Finished the sampling")
    chosen_samples_ids = BitMask.from_hex(new_tag.bit_mask_data).to_indices()
    chosen_filenames = [api_workflow.filenames[i] for i in chosen_samples_ids]

    print(f'chosen_filenames: {chosen_filenames}')
    print("Finished the sampling")


if __name__ == "__main__":
    path_to_dataset = "/Users/malteebnerlightly/Documents/datasets/clothing-dataset-small-master/test/dress"
    token = os.getenv("TOKEN")
    dataset_id = "6023faa0c9d7110032b9cbcb"
    for i in range(2):
        print(f"ITERATION {i}:")
        t_est_unmocked_complete_workflow(path_to_dataset, token, dataset_id)
        print("")
