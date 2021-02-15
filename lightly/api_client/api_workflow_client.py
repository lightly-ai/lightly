from typing import *

from lightly.data.dataset import LightlyDataset
from lightly.api.upload import upload_file_with_signed_url, upload_images_from_folder, upload_dataset
from lightly.openapi_generated.swagger_client.api.embeddings_api import EmbeddingsApi
from lightly.openapi_generated.swagger_client.api.jobs_api import JobsApi
from lightly.openapi_generated.swagger_client.api.mappings_api import MappingsApi
from lightly.openapi_generated.swagger_client.api.samplings_api import SamplingsApi
from lightly.openapi_generated.swagger_client.api.tags_api import TagsApi
from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.configuration import Configuration
from lightly.openapi_generated.swagger_client.models.dataset_embedding_data import DatasetEmbeddingData
from lightly.openapi_generated.swagger_client.models.write_csv_url_data import WriteCSVUrlData
from .api_workflow_upload_embeddings import UploadEmbeddingsMixin


class ApiWorkflowClient(UploadEmbeddingsMixin):
    def __init__(self, host: str, token: str, dataset_id: str, embedding_id: str = None):

        configuration = Configuration()
        configuration.host = host
        configuration.api_key = {'token': token}
        api_client = ApiClient(configuration=configuration)
        self.api_client = api_client

        self.dataset_id = dataset_id
        if embedding_id is not None:
            self.embedding_id = embedding_id

        self.samplings_api = SamplingsApi(api_client=self.api_client)
        self.jobs_api = JobsApi(api_client=self.api_client)
        self.tags_api = TagsApi(api_client=self.api_client)
        self.embeddings_api = EmbeddingsApi(api_client=api_client)
        self.mappings_api = MappingsApi(api_client=api_client)

    def order_list_by_filenames(self, filenames_for_list: List[str], list_to_order: List[object]) -> List[object]:
        assert len(filenames_for_list) == len(list_to_order)
        dict_by_filenames = dict([(filename, element) for filename, element in zip(filenames_for_list, list_to_order)])
        list_ordered = [dict_by_filenames[filename] for filename in self.filenames_on_server
                        if filename in filenames_for_list]
        return list_ordered

    @property
    def filenames_on_server(self):
        if not hasattr(self, "_filenames"):
            self._filenames_on_server = self.mappings_api. \
                get_sample_mappings_by_dataset_id(dataset_id=self.dataset_id, field="fileName")
        return self._filenames_on_server

    def upload_dataset(self, input: Union[str, LightlyDataset], **kwargs):
        if isinstance(input, str):
            path_to_dataset = input
            upload_images_from_folder(path_to_dataset, self.dataset_id, self.token, **kwargs)
        elif isinstance(input, LightlyDataset):
            dataset = input
            upload_dataset(dataset, self.dataset_id, self.tags_api, **kwargs)
        else:
            raise ValueError(f"input must either be a LightlyDataset or the path to the dataset as str, "
                             f"but is of type {type(input)}")
