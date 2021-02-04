import tempfile
import os

from lightly.api.api_workflow import ApiWorkflow

from lightly.data import LightlyDataset
from lightly.api.upload import upload_dataset

from lightly.core import train_model_and_embed_images
from lightly.openapi_generated.swagger_client import InitialTagCreateRequest
from lightly.utils import save_embeddings

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.api.bitmask import BitMask


def t_est_unmocked_complete_workflow(path_to_dataset: str, token: str, dataset_id: str):

    # define the api_client and api_workflow
    api_workflow = ApiWorkflow(host="https://api-dev.lightly.ai", token=token, dataset_id=dataset_id)

    # upload the images to the dataset and create the initial tag
    if len(api_workflow.tags_api.get_tags_by_dataset_id(dataset_id=dataset_id)) == 0:
        dataset = LightlyDataset(input_dir=path_to_dataset)
        upload_dataset(dataset=dataset, dataset_id=dataset_id, token=token, max_workers=1)
        initial_tag_create_request = InitialTagCreateRequest()
        api_workflow.tags_api.create_initial_tag_by_dataset_id(body=initial_tag_create_request, dataset_id=dataset_id)

    # calculate and save the embeddings
    path_to_embeddings_csv = f"{path_to_dataset}/embeddings.csv"
    if not os.path.isfile(path_to_embeddings_csv):
        embeddings, labels, filenames = train_model_and_embed_images(input_dir=path_to_dataset)
        save_embeddings(path_to_embeddings_csv, embeddings, labels, filenames)

    # upload the embeddings
    api_workflow.upload_embeddings(path_to_embeddings_csv=path_to_embeddings_csv, name="embedding_1")

    # perform_a_sampling
    sampler_config = SamplerConfig()
    new_tag = api_workflow.sampling(sampler_config=sampler_config)
    chosen_samples_ids = BitMask.from_bin(new_tag.bit_mask_data)

    print(new_tag)
    print(f'chosen_sample_ids: {chosen_samples_ids}')


if __name__ == "__main__":
    path_to_dataset = "/Users/malteebnerlightly/Documents/datasets/clothing-dataset-small-master/test"
    token = "f9b60358d529bdd824e3c2df"
    dataset_id = "601c014812cb7f0032875f4f"
    t_est_unmocked_complete_workflow(path_to_dataset, token, dataset_id)
