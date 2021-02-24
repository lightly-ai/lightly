import os
from typing import *

from lightly.api.api_workflow_upload_embeddings import _UploadEmbeddingsMixin
from lightly.api.api_workflow_sampling import _SamplingMixin
from lightly.data.dataset import LightlyDataset
from lightly.api.upload import upload_images_from_folder, upload_dataset
from lightly.openapi_generated.swagger_client import TagData, ScoresApi
from lightly.openapi_generated.swagger_client.api.embeddings_api import EmbeddingsApi
from lightly.openapi_generated.swagger_client.api.jobs_api import JobsApi
from lightly.openapi_generated.swagger_client.api.mappings_api import MappingsApi
from lightly.openapi_generated.swagger_client.api.samplings_api import SamplingsApi
from lightly.openapi_generated.swagger_client.api.tags_api import TagsApi
from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.configuration import Configuration


class ApiWorkflowClient(_UploadEmbeddingsMixin, _SamplingMixin):
    """Provides a uniform interface to communicate with the api and run workflows including multiple API calls

    Args:
        host:
            the url of the server, e.g. https://api-dev.lightly.ai
        token:
            the token of the user, provided in webapp
        dataset_id:
            the id of the dataset, provided in webapp
        embedding_id:
            the id of the embedding to use. If it is not set, but used by a workflow, the newest embedding is taken by default
    """

    def __init__(self, host: str, token: str, dataset_id: str, embedding_id: str = None):

        configuration = Configuration()
        configuration.host = host
        configuration.api_key = {'token': token}
        api_client = ApiClient(configuration=configuration)
        self.api_client = api_client

        os.environ["LIGHTLY_SERVER_LOCATION"] = host
        self.token = token
        self.dataset_id = dataset_id
        if embedding_id is not None:
            self.embedding_id = embedding_id

        self.samplings_api = SamplingsApi(api_client=self.api_client)
        self.jobs_api = JobsApi(api_client=self.api_client)
        self.tags_api = TagsApi(api_client=self.api_client)
        self.embeddings_api = EmbeddingsApi(api_client=api_client)
        self.mappings_api = MappingsApi(api_client=api_client)
        self.scores_api = ScoresApi(api_client=api_client)

    def _get_all_tags(self) -> List[TagData]:
        return self.tags_api.get_tags_by_dataset_id(self.dataset_id)

    def _order_list_by_filenames(self, filenames_for_list: List[str], list_to_order: List[object]) -> List[object]:
        """Orders a list such that it is in the order of the filenames specified on the server.

        Args:
            filenames_for_list:
                The filenames of samples in a specific order
            list_to_order:
                Some values belonging to the samples

        Returns:
            The list reorderd. The same reorder applied on the filenames_for_list
            would put them in the order of the filenames in self.filenames_on_server

        """
        assert len(filenames_for_list) == len(list_to_order)
        dict_by_filenames = dict(zip(filenames_for_list, list_to_order))
        list_ordered = [dict_by_filenames[filename] for filename in self.filenames_on_server
                        if filename in filenames_for_list]
        return list_ordered

    @property
    def filenames_on_server(self):
        if not hasattr(self, "_filenames_on_server"):
            self._filenames_on_server = self.mappings_api. \
                get_sample_mappings_by_dataset_id(dataset_id=self.dataset_id, field="fileName")
        return self._filenames_on_server

    def upload_dataset(self, input: Union[str, LightlyDataset], **kwargs):
        """Uploads a dataset to the server and creates the initial tag.

        Args:
            input:
                one of the following:
                    - the path to the dataset, e.g. "path/to/dataset"
                    - the dataset in form of a LightlyDataset
            **kwargs:
                see specification of the called functions
        """
        if isinstance(input, str):
            path_to_dataset = input
            upload_images_from_folder(path_to_dataset, self.dataset_id, self.token, **kwargs)
        elif isinstance(input, LightlyDataset):
            dataset = input
            upload_dataset(dataset, self.dataset_id, self.token, **kwargs)
        else:
            raise ValueError(f"input must either be a LightlyDataset or the path to the dataset as str, "
                             f"but is of type {type(input)}")
