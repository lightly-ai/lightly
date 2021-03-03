import os
from typing import *

from lightly.openapi_generated.swagger_client.api.samples_api import SamplesApi

from lightly.api.utils import put_request, getenv

from lightly.api.api_workflow_upload_dataset import _UploadDatasetMixin
from lightly.api.api_workflow_upload_embeddings import _UploadEmbeddingsMixin
from lightly.api.api_workflow_sampling import _SamplingMixin
from lightly.openapi_generated.swagger_client import TagData, ScoresApi
from lightly.openapi_generated.swagger_client.api.embeddings_api import EmbeddingsApi
from lightly.openapi_generated.swagger_client.api.jobs_api import JobsApi
from lightly.openapi_generated.swagger_client.api.mappings_api import MappingsApi
from lightly.openapi_generated.swagger_client.api.samplings_api import SamplingsApi
from lightly.openapi_generated.swagger_client.api.tags_api import TagsApi
from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.configuration import Configuration


class ApiWorkflowClient(_UploadEmbeddingsMixin, _SamplingMixin, _UploadDatasetMixin):
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

    def __init__(self, token: str, dataset_id: str, embedding_id: str = None):

        configuration = Configuration()
        configuration.host = getenv('LIGHTLY_SERVER_LOCATION', 'https://api.lightly.ai')
        configuration.api_key = {'token': token}
        api_client = ApiClient(configuration=configuration)
        self.api_client = api_client

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
        self.samples_api = SamplesApi(api_client=api_client)

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

    def upload_file_with_signed_url(self, file, signed_write_url: str):
        response = put_request(signed_write_url, data=file)
        file.close()
        return response
