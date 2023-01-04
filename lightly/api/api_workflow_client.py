import warnings
from io import IOBase
from typing import *
import platform
import os

import requests

from lightly.api.api_workflow_predictions import _PredictionsMixin
from lightly.api.api_workflow_tags import _TagsMixin
from requests import Response

from lightly.__init__ import __version__
from lightly.api.api_workflow_collaboration import _CollaborationMixin
from lightly.api.api_workflow_compute_worker import _ComputeWorkerMixin
from lightly.api.api_workflow_datasets import _DatasetsMixin
from lightly.api.api_workflow_datasources import _DatasourcesMixin
from lightly.api.api_workflow_download_dataset import _DownloadDatasetMixin
from lightly.api.api_workflow_selection import _SelectionMixin
from lightly.api.api_workflow_upload_dataset import _UploadDatasetMixin
from lightly.api.api_workflow_upload_embeddings import _UploadEmbeddingsMixin
from lightly.api.api_workflow_upload_metadata import _UploadCustomMetadataMixin
from lightly.api.utils import DatasourceType, get_signed_url_destination, get_api_client_configuration
from lightly.api.version_checking import is_compatible_version, LightlyAPITimeoutException
from lightly.openapi_generated.swagger_client.api.collaboration_api import CollaborationApi
from lightly.openapi_generated.swagger_client import ScoresApi, QuotaApi, MetaDataConfigurationsApi, PredictionsApi
from lightly.openapi_generated.swagger_client.api.datasets_api import \
    DatasetsApi
from lightly.openapi_generated.swagger_client.api.datasources_api import \
    DatasourcesApi
from lightly.openapi_generated.swagger_client.api.docker_api import DockerApi
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
from lightly.openapi_generated.swagger_client.models.dataset_data import \
    DatasetData
from lightly.utils.reordering import sort_items_by_keys

# Env variable for server side encryption on S3
LIGHTLY_S3_SSE_KMS_KEY = 'LIGHTLY_S3_SSE_KMS_KEY' 

class ApiWorkflowClient(_UploadEmbeddingsMixin,
                        _SelectionMixin,
                        _UploadDatasetMixin,
                        _DownloadDatasetMixin,
                        _DatasetsMixin,
                        _UploadCustomMetadataMixin,
                        _TagsMixin,
                        _DatasourcesMixin,
                        _ComputeWorkerMixin,
                        _CollaborationMixin,
                        _PredictionsMixin,
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

    def __init__(
        self,
        token: Optional[str] = None,
        dataset_id: Optional[str] = None,
        embedding_id: Optional[str] = None
    ):

        try:
            if not is_compatible_version(__version__):
                warnings.warn(
                    UserWarning((f"Incompatible version of lightly pip package. "
                                f"Please upgrade to the latest version "
                                f"to be able to access the api.")
                    )
                )
        except LightlyAPITimeoutException:
            pass

        configuration = get_api_client_configuration(token=token)
        self.api_client = ApiClient(configuration=configuration)
        self.user_agent = f"Lightly/{__version__}/python ({platform.platform()})"
        self.set_request_timeout(DEFAULT_API_TIMEOUT)

        self.token = configuration.api_key["token"]
        if dataset_id is not None:
            self._dataset_id = dataset_id
        if embedding_id is not None:
            self.embedding_id = embedding_id

        self._collaboration_api = CollaborationApi(api_client=self.api_client)
        self._compute_worker_api = DockerApi(api_client=self.api_client)
        self._datasets_api = DatasetsApi(api_client=self.api_client)
        self._datasources_api = DatasourcesApi(api_client=self.api_client)
        self._selection_api = SamplingsApi(api_client=self.api_client)
        self._jobs_api = JobsApi(api_client=self.api_client)
        self._tags_api = TagsApi(api_client=self.api_client)
        self._embeddings_api = EmbeddingsApi(api_client=self.api_client)
        self._mappings_api = MappingsApi(api_client=self.api_client)
        self._scores_api = ScoresApi(api_client=self.api_client)
        self._samples_api = SamplesApi(api_client=self.api_client)
        self._quota_api = QuotaApi(api_client=self.api_client)
        self._metadata_configurations_api = \
            MetaDataConfigurationsApi(api_client=self.api_client)
        self._predictions_api = PredictionsApi(api_client=self.api_client)

    @property
    def dataset_id(self) -> str:
        '''The current dataset_id.

        If the dataset_id is set, it is returned.
        If it is not set, then the dataset_id of the last modified dataset is selected.
        ''' 
        try:
            return self._dataset_id
        except AttributeError:
            all_datasets: List[DatasetData] = self.get_datasets()
            datasets_sorted = sorted(all_datasets, key=lambda dataset: dataset.last_modified_at)
            last_modified_dataset = datasets_sorted[-1]
            self._dataset_id = last_modified_dataset.id
            warnings.warn(UserWarning(f"Dataset has not been specified, "
                                      f"taking the last modified dataset {last_modified_dataset.name} as default dataset."))
            return self._dataset_id

    @dataset_id.setter
    def dataset_id(self, dataset_id: str):
        """Sets the current dataset id for the client.
        
        Args:
            dataset_id:
                The new dataset id.

        Raises:
            ValueError if the dataset id does not exist.
        """
        if not self.dataset_exists(dataset_id):
            raise ValueError(
                f"A dataset with the id {dataset_id} does not exist on the web"
                f"platform."
            )
        self._dataset_id = dataset_id
        

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
        filenames_on_server = self.get_filenames()
        list_ordered = sort_items_by_keys(
            filenames_for_list, list_to_order, filenames_on_server
        )
        return list_ordered

    def get_filenames(self) -> List[str]:
        """Downloads the list of filenames from the server.

        This is an expensive operation, especially for large datasets.
        """
        filenames_on_server = self._mappings_api. \
            get_sample_mappings_by_dataset_id(dataset_id=self.dataset_id, field="fileName")
        return filenames_on_server

    def upload_file_with_signed_url(
        self,
        file: IOBase,
        signed_write_url: str,
        headers: Optional[Dict] = None,
        session: Optional[requests.Session] = None,
    ) -> Response:
        """Uploads a file to a url via a put request.

        Args:
            file:
                The file to upload.
            signed_write_url:
                The url to upload the file to. As no authorization is used,
                the url must be a signed write url.
            headers:
                Specific headers for the request.
            session:
                Optional requests session used to upload the file.

        Returns:
            The response of the put request, usually a 200 for the success case.

        """

        # check to see if server side encryption for S3 is desired
        # see https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingServerSideEncryption.html
        # see https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingKMSEncryption.html
        lightly_s3_sse_kms_key = os.environ.get(LIGHTLY_S3_SSE_KMS_KEY, '').strip()
        # Only set s3 related headers when we are talking with s3
        if get_signed_url_destination(signed_write_url)==DatasourceType.S3 and lightly_s3_sse_kms_key:
            if headers is None:
                headers = {}
            # don't override previously set SSE
            if 'x-amz-server-side-encryption' not in headers:
                if lightly_s3_sse_kms_key.lower() == 'true':
                    # enable SSE with the key of amazon
                    headers['x-amz-server-side-encryption'] = 'AES256'
                else:
                    # enable SSE with specific customer KMS key
                    headers['x-amz-server-side-encryption'] = 'aws:kms'
                    headers['x-amz-server-side-encryption-aws-kms-key-id'] = lightly_s3_sse_kms_key

        # start requests session and make put request
        sess = session or requests
        if headers is not None:
            response = sess.put(signed_write_url, data=file, headers=headers)
        else:
            response = sess.put(signed_write_url, data=file)
        response.raise_for_status()
        return response

    def set_request_timeout(self, timeout: Union[int, Tuple[int, int]]):
        """Sets a default timeout for all api requests.

        Args:
            timeout:
                Timeout in seconds. Is either a single total_timeout value or a
                (connect_timeout, read_timeout) tuple. 
                See https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html?highlight=timeout#urllib3.util.Timeout
                for details on the different values.
        """
        set_api_client_request_timeout(client=self.api_client, timeout=timeout)


DEFAULT_API_TIMEOUT = 60 * 3 # seconds

def set_api_client_request_timeout(
    client: ApiClient, 
    timeout: Union[int, Tuple[int, int]] = DEFAULT_API_TIMEOUT,
):
    """Sets a default timeout for all requests with the client.

    This function patches the request method of the api client. This is 
    necessary because the swagger api client does not respect any timeouts 
    configured by urllib3. Instead it expects a timeout to be passed with every
    request. Code here: https://github.com/lightly-ai/lightly/blob/ffbd32fe82f76b37c8ac497640355314474bfc3b/lightly/openapi_generated/swagger_client/rest.py#L141-L148

    Args:
        client:
            Api client on which the timeout is applied.
        timeout:
            Timeout in seconds. Is either a single total_timeout value or a
            (connect_timeout, read_timeout) tuple. 
            See https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html?highlight=timeout#urllib3.util.Timeout
            for details on the different values.

    """
    request_fn = client.rest_client.request

    def new_request_fn(*args, **kwargs):
        request_timeout = kwargs['_request_timeout']
        if request_timeout is None:
            kwargs['_request_timeout'] = timeout
        return request_fn(*args, **kwargs)

    client.rest_client.request = new_request_fn
