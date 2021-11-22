import warnings
from io import IOBase
from typing import *

import requests
from lightly.api.api_workflow_tags import _TagsMixin
from requests import Response

from lightly.__init__ import __version__
from lightly.api.api_workflow_datasets import _DatasetsMixin
from lightly.api.api_workflow_download_dataset import _DownloadDatasetMixin
from lightly.api.api_workflow_sampling import _SamplingMixin
from lightly.api.api_workflow_upload_dataset import _UploadDatasetMixin
from lightly.api.api_workflow_upload_embeddings import _UploadEmbeddingsMixin
from lightly.api.api_workflow_upload_metadata import _UploadCustomMetadataMixin
from lightly.api.bitmask import BitMask
from lightly.api.utils import getenv
from lightly.api.version_checking import get_minimum_compatible_version, \
    version_compare
from lightly.openapi_generated.swagger_client import TagData, ScoresApi, \
    QuotaApi, TagArithmeticsRequest, TagArithmeticsOperation, \
    TagBitMaskResponse
from lightly.openapi_generated.swagger_client.api.datasets_api import \
    DatasetsApi
from lightly.openapi_generated.swagger_client.api.embeddings_api import \
    EmbeddingsApi
from lightly.openapi_generated.swagger_client.api.jobs_api import JobsApi
from lightly.openapi_generated.swagger_client.api.mappings_api import \
    MappingsApi
from lightly.openapi_generated.swagger_client.api.samples_api import SamplesApi
from lightly.openapi_generated.swagger_client.api.samplings_api import \
    SamplingsApi
from lightly.openapi_generated.swagger_client.api.tags_api import TagsApi
from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.configuration import \
    Configuration
from lightly.openapi_generated.swagger_client.models.dataset_data import \
    DatasetData


class ApiWorkflowClient(_UploadEmbeddingsMixin,
                        _SamplingMixin,
                        _UploadDatasetMixin,
                        _DownloadDatasetMixin,
                        _DatasetsMixin,
                        _UploadCustomMetadataMixin,
                        _TagsMixin
                        ):
    """Provides a uniform interface to communicate with the api 
    
    The APIWorkflowClient is used to communicaate with the Lightly API. The client
    can run also more complex workflows which include multiple API calls at once.
    
    The client can be used in combination with the active learning agent. 

    Args:
        token:
            the token of the user, provided in webapp
        dataset_id:
            the id of the dataset, provided in webapp. \
            If it is not set, but used by a workflow, \
            the last modfied dataset is taken by default.
        embedding_id:
            the id of the embedding to use. If it is not set, \
            but used by a workflow, the newest embedding is taken by default
    """

    def __init__(self, token: str, dataset_id: str = None, embedding_id: str = None):

        self.check_version_compatibility()

        configuration = Configuration()
        configuration.host = getenv('LIGHTLY_SERVER_LOCATION', 'https://api.lightly.ai')
        configuration.api_key = {'token': token}
        api_client = ApiClient(configuration=configuration)
        self.api_client = api_client

        self.token = token
        if dataset_id is not None:
            self._dataset_id = dataset_id
        if embedding_id is not None:
            self.embedding_id = embedding_id

        self._datasets_api = DatasetsApi(api_client=self.api_client)
        self._samplings_api = SamplingsApi(api_client=self.api_client)
        self._jobs_api = JobsApi(api_client=self.api_client)
        self._tags_api = TagsApi(api_client=self.api_client)
        self._embeddings_api = EmbeddingsApi(api_client=api_client)
        self._mappings_api = MappingsApi(api_client=api_client)
        self._scores_api = ScoresApi(api_client=api_client)
        self._samples_api = SamplesApi(api_client=api_client)
        self._quota_api = QuotaApi(api_client=api_client)

    def check_version_compatibility(self):
        minimum_version = get_minimum_compatible_version()
        if version_compare(__version__, minimum_version) < 0:
            raise ValueError(f"Incompatible Version of lightly pip package. "
                             f"Please upgrade to at least version {minimum_version} "
                             f"to be able to access the api and webapp")

    @property
    def dataset_id(self) -> str:
        '''The current dataset_id.

        If the dataset_id is set, it is returned.
        If it is not set, then the dataset_id of the last modified dataset is selected.
        ''' 
        try:
            return self._dataset_id
        except AttributeError:
            all_datasets: List[DatasetData] = self._datasets_api.get_datasets()
            datasets_sorted = sorted(all_datasets, key=lambda dataset: dataset.last_modified_at)
            last_modified_dataset = datasets_sorted[-1]
            self._dataset_id = last_modified_dataset.id
            warnings.warn(UserWarning(f"Dataset has not been specified, "
                                      f"taking the last modified dataset {last_modified_dataset.name} as default dataset."))
            return self._dataset_id

    def _order_list_by_filenames(
            self, filenames_for_list: List[str],
            list_to_order: List[object]
    ) -> List[object]:
        """Orders a list such that it is in the order of the filenames specified on the server.

        Args:
            filenames_for_list:
                The filenames of samples in a specific order
            list_to_order:
                Some values belonging to the samples

        Returns:
            The list reordered.
            The same reorder applied on the filenames_for_list would put them
            in the order of the filenames in self.filenames_on_server.
            every filename in self.filenames_on_server must be in the
            filenames_for_list.

        """
        filenames_on_server = self.download_filenames_from_server()
        if len(filenames_for_list) != len(list_to_order) or \
                len(filenames_for_list) != len(filenames_on_server):
            raise ValueError(
                f"All inputs (filenames_for_list,  list_to_order and "
                f"self.filenames_on_server) must have the same length, "
                f"but their lengths are: ({len(filenames_for_list)},"
                f"{len(list_to_order)} and {len(filenames_on_server)})."
            )
        dict_by_filenames = dict(zip(filenames_for_list, list_to_order))
        list_ordered = [dict_by_filenames[filename] for filename in filenames_on_server]
        return list_ordered

    def download_filenames_from_server(self) -> List[str]:
        """Downloads the list of filenames from the server.

        This is an expensive operation, especially for large datasets.
        """
        filenames_on_server = self._mappings_api. \
            get_sample_mappings_by_dataset_id(dataset_id=self.dataset_id, field="fileName")
        return filenames_on_server

    def get_tag_and_filenames(
            self,
            tag_name: str = None,
            tag_id: str = None,
            filenames_on_server: List[str] = None,
            exclude_parent_tag: bool = False
    ) -> Tuple[TagData, List[str]]:
        """Gets the TagData of a tag and the filenames in it

        The tag_name or the tag_id must be passed.
        The tag_name overwrites the tag_id.
        Args:
            tag_name:
                The name of the tag.
            tag_id:
                The id of the tag.
            filenames_on_server:
                List of all filenames on the server. If they are not given,
                they need to be download newly, which is quite expensive.
            exclude_parent_tag:
                Excludes the parent tag in the returned filenames.

        Returns:
            tag_data:
                The tag_data, raw from the API
            filenames_tag:
                The filenames of all samples in the tag.

        """
        tag_name_id_dict = {tag.name: tag.id for tag in self._get_all_tags()}
        if tag_name:
            tag_name_id_dict = {
                tag.name: tag.id for tag in self._get_all_tags()
            }
            tag_id = tag_name_id_dict.get(tag_name, None)
            if tag_id is None:
                raise ValueError(f'Your tag_name is invalid: {tag_name}.')
        elif tag_id is None or tag_id not in tag_name_id_dict.values():
            raise ValueError(f'Your tag_id is invalid: {tag_id}')
        tag_data = self._tags_api.get_tag_by_tag_id(self.dataset_id, tag_id)

        if exclude_parent_tag:
            parent_tag_id = tag_data.prev_tag_id
            tag_arithmetics_request = TagArithmeticsRequest(
                tag_id1=tag_data.id, tag_id2=parent_tag_id,
                operation=TagArithmeticsOperation.DIFFERENCE)
            bit_mask_response: TagBitMaskResponse = \
                self._tags_api.perform_tag_arithmetics(
                    body=tag_arithmetics_request, dataset_id=self.dataset_id
                )
            bit_mask_data = bit_mask_response.bit_mask_data
        else:
            bit_mask_data = tag_data.bit_mask_data

        if not filenames_on_server:
            filenames_on_server = self.download_filenames_from_server()

        chosen_samples_ids = BitMask.from_hex(bit_mask_data).to_indices()
        filenames_tag = [filenames_on_server[i] for i in chosen_samples_ids]

        return tag_data, filenames_tag

    def upload_file_with_signed_url(self,
                                    file: IOBase,
                                    signed_write_url: str) -> Response:
        """Uploads a file to a url via a put request.

        Args:
            file:
                The file to upload.
            signed_write_url:
                The url to upload the file to. As no authorization is used,
                the url must be a signed write url.

        Returns:
            The response of the put request, usually a 200 for the success case.

        """

        response = requests.put(signed_write_url, data=file)

        if response.status_code != 200:
            msg = f'Failed PUT request to {signed_write_url} with status_code'
            msg += f'{response.status__code}!'
            raise RuntimeError(msg)

        return response
