import csv
import io
import tempfile
import unittest
from io import IOBase

import numpy as np

from lightly.openapi_generated.swagger_client.models.tag_creator import TagCreator

from requests import Response

from lightly.openapi_generated.swagger_client.models.dataset_create_request import DatasetCreateRequest

from lightly.openapi_generated.swagger_client.models.dataset_data import DatasetData

from lightly.openapi_generated.swagger_client.api.datasets_api import DatasetsApi

import lightly

from lightly.api.api_workflow_client import ApiWorkflowClient

from typing import *

from lightly.openapi_generated.swagger_client import ScoresApi, \
    CreateEntityResponse, SamplesApi, SampleCreateRequest, \
    InitialTagCreateRequest, ApiClient, VersioningApi, QuotaApi, \
    TagArithmeticsRequest, TagBitMaskResponse, SampleWriteUrls, SampleData, \
    Trigger2dEmbeddingJobRequest
from lightly.openapi_generated.swagger_client.api.embeddings_api import EmbeddingsApi
from lightly.openapi_generated.swagger_client.api.jobs_api import JobsApi
from lightly.openapi_generated.swagger_client.api.mappings_api import MappingsApi
from lightly.openapi_generated.swagger_client.api.samplings_api import SamplingsApi
from lightly.openapi_generated.swagger_client.api.tags_api import TagsApi
from lightly.openapi_generated.swagger_client.models.async_task_data import AsyncTaskData
from lightly.openapi_generated.swagger_client.models.dataset_embedding_data import DatasetEmbeddingData
from lightly.openapi_generated.swagger_client.models.job_result_type import JobResultType
from lightly.openapi_generated.swagger_client.models.job_state import JobState
from lightly.openapi_generated.swagger_client.models.job_status_data import JobStatusData
from lightly.openapi_generated.swagger_client.models.job_status_data_result import JobStatusDataResult
from lightly.openapi_generated.swagger_client.models.sampling_create_request import SamplingCreateRequest
from lightly.openapi_generated.swagger_client.models.tag_data import TagData
from lightly.openapi_generated.swagger_client.models.write_csv_url_data import WriteCSVUrlData


def _check_dataset_id(dataset_id: str):
    assert isinstance(dataset_id, str)
    assert len(dataset_id) > 0


N_FILES_ON_SERVER = 100


class MockedEmbeddingsApi(EmbeddingsApi):
    def __init__(self, api_client):
        EmbeddingsApi.__init__(self, api_client=api_client)
        self.embeddings = [
            DatasetEmbeddingData(
                id='embedding_id_xyz',
                name='embedding_name_xxyyzz',
                is_processed=True,
                created_at=0,
            ),
            DatasetEmbeddingData(
                id='embedding_id_xyz_2',
                name='default',
                is_processed=True,
                created_at=0,
            )
        
        ]

    def get_embeddings_csv_write_url_by_id(self, dataset_id: str, **kwargs):
        _check_dataset_id(dataset_id)
        assert isinstance(dataset_id, str)
        response_ = WriteCSVUrlData(signed_write_url="signed_write_url_valid", embedding_id="embedding_id_xyz")
        return response_

    def get_embeddings_by_dataset_id(self, dataset_id, **kwargs) -> List[DatasetEmbeddingData]:
        _check_dataset_id(dataset_id)
        assert isinstance(dataset_id, str)
        return self.embeddings

    def trigger2d_embeddings_job(self, body, dataset_id, embedding_id, **kwargs):
        _check_dataset_id(dataset_id)
        assert isinstance(body, Trigger2dEmbeddingJobRequest)

    def get_embeddings_csv_read_url_by_id(self, dataset_id, embedding_id, **kwargs):
        _check_dataset_id(dataset_id)
        return 'https://my-embedding-read-url.com'


class MockedSamplingsApi(SamplingsApi):
    def trigger_sampling_by_id(self, body: SamplingCreateRequest, dataset_id, embedding_id, **kwargs):
        _check_dataset_id(dataset_id)
        assert isinstance(body, SamplingCreateRequest)
        assert isinstance(dataset_id, str)
        assert isinstance(embedding_id, str)
        response_ = AsyncTaskData(job_id="155")
        return response_


class MockedJobsApi(JobsApi):
    def __init__(self, *args, **kwargs):
        self.no_calls = 0
        JobsApi.__init__(self, *args, **kwargs)

    def get_job_status_by_id(self, job_id, **kwargs):
        assert isinstance(job_id, str)
        self.no_calls += 1
        if self.no_calls > 3:
            result = JobStatusDataResult(type=JobResultType.SAMPLING, data="sampling_tag_id_xyz")
            response_ = JobStatusData(id="id_", status=JobState.FINISHED, wait_time_till_next_poll=0,
                                      created_at=1234, finished_at=1357, result=result)
        else:
            result = None
            response_ = JobStatusData(id="id_", status=JobState.RUNNING, wait_time_till_next_poll=0.001,
                                      created_at=1234, result=result)
        return response_


class MockedTagsApi(TagsApi):
    def create_initial_tag_by_dataset_id(self, body, dataset_id, **kwargs):
        _check_dataset_id(dataset_id)
        assert isinstance(body, InitialTagCreateRequest)
        assert isinstance(dataset_id, str)
        response_ = CreateEntityResponse(id="xyz")
        return response_

    def get_tag_by_tag_id(self, dataset_id, tag_id, **kwargs):
        _check_dataset_id(dataset_id)
        assert isinstance(dataset_id, str)
        assert isinstance(tag_id, str)
        response_ = TagData(id=tag_id, dataset_id=dataset_id, prev_tag_id="initial-tag", bit_mask_data="0x80bda23e9",
                            name='second-tag', tot_size=15, created_at=1577836800, changes=dict())
        return response_

    def get_tags_by_dataset_id(self, dataset_id, **kwargs):
        _check_dataset_id(dataset_id)
        if dataset_id == 'xyz-no-tags':
            return []
        tag_1 = TagData(id='inital_tag_id', dataset_id=dataset_id, prev_tag_id=None,
                        bit_mask_data="0xF", name='initial-tag', tot_size=4,
                        created_at=1577836800, changes=dict())
        tag_2 = TagData(id='query_tag_id_xyz', dataset_id=dataset_id, prev_tag_id="initial-tag",
                        bit_mask_data="0xF", name='query_tag_name_xyz', tot_size=4,
                        created_at=1577836800, changes=dict())
        tag_3 = TagData(id='preselected_tag_id_xyz', dataset_id=dataset_id, prev_tag_id="initial-tag",
                        bit_mask_data="0x1", name='preselected_tag_name_xyz', tot_size=4,
                        created_at=1577836800, changes=dict())
        tag_4 = TagData(id='sampled_tag_xyz', dataset_id=dataset_id, prev_tag_id="preselected_tag_id_xyz",
                        bit_mask_data="0x3", name='sampled_tag_xyz', tot_size=4,
                        created_at=1577836800, changes=dict())
        tag_5 = TagData(id='tag_with_integer_name', dataset_id=dataset_id, prev_tag_id=None,
                        bit_mask_data='0x1', name='1000', tot_size=4,
                        created_at=1577836800, changes=dict())
        tags = [tag_1, tag_2, tag_3, tag_4, tag_5]
        no_tags_to_return = getattr(self, "no_tags", 5)
        tags = tags[:no_tags_to_return]
        return tags

    def perform_tag_arithmetics(self, body: TagArithmeticsRequest, dataset_id, **kwargs):
        _check_dataset_id(dataset_id)
        return TagBitMaskResponse(bit_mask_data="0x2")

    def upsize_tags_by_dataset_id(self, body, dataset_id, **kwargs):
        _check_dataset_id(dataset_id)
        assert body.upsize_tag_creator == TagCreator.USER_PIP


class MockedScoresApi(ScoresApi):
    def create_or_update_active_learning_score_by_tag_id(self, body, dataset_id, tag_id, **kwargs) -> \
            CreateEntityResponse:
        _check_dataset_id(dataset_id)
        if len(body.scores) > 0 and not isinstance(body.scores[0], float):
            raise AttributeError
        response_ = CreateEntityResponse(id="sampled_tag_id_xyz")
        return response_


class MockedMappingsApi(MappingsApi):
    def __init__(self, samples_api, *args, **kwargs):
        self.samples_api = samples_api
        MappingsApi.__init__(self, *args, **kwargs)

        self.n_samples = N_FILES_ON_SERVER
        sample_names = [f'img_{i}.jpg' for i in range(self.n_samples)]
        sample_names.reverse()
        self.sample_names = sample_names
        

    def get_sample_mappings_by_dataset_id(self, dataset_id, field, **kwargs):
        if dataset_id == 'xyz-no-tags':
            return []
        return self.sample_names[:self.n_samples]


class MockedSamplesApi(SamplesApi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_create_requests: List[SampleCreateRequest] = []

    def get_samples_by_dataset_id(self, dataset_id, **kwargs) -> List[SampleData]:
        samples = []
        for i, body in enumerate(self.sample_create_requests):
            sample = SampleData(id=f'{i}_xyz', dataset_id='dataset_id_xyz', file_name=body.file_name)
            samples.append(sample)
        return samples

    def create_sample_by_dataset_id(self, body, dataset_id, **kwargs):
        _check_dataset_id(dataset_id)
        assert isinstance(body, SampleCreateRequest)
        response_ = CreateEntityResponse(id="xyz")
        self.sample_create_requests.append(body)
        return response_

    def get_sample_image_write_url_by_id(self, dataset_id, sample_id, is_thumbnail, **kwargs):
        _check_dataset_id(dataset_id)
        url = f"{sample_id}_write_url"
        return url

    def get_sample_image_read_url_by_id(self, dataset_id, sample_id, type, **kwargs):
        _check_dataset_id(dataset_id)
        url = f"{sample_id}_write_url"
        return url

    def get_sample_image_write_urls_by_id(self, dataset_id, sample_id, **kwargs) -> SampleWriteUrls:
        _check_dataset_id(dataset_id)
        thumb_url = f"{sample_id}_thumb_write_url"
        full_url = f"{sample_id}_full_write_url"
        ret = SampleWriteUrls(full=full_url, thumb=thumb_url)
        return ret


class MockedDatasetsApi(DatasetsApi):
    def __init__(self, api_client):
        no_datasets = 3
        self.default_datasets = [DatasetData(name=f"dataset_{i}", id=f"dataset_{i}_id", last_modified_at=i,
                                             type="", img_type="full", size_in_bytes=-1, n_samples=-1, created_at=-1)
                                 for i in range(no_datasets)]
        self.reset()

    def reset(self):
        self.datasets = self.default_datasets

    def get_datasets(self, **kwargs):
        return self.datasets

    def create_dataset(self, body: DatasetCreateRequest, **kwargs):
        assert isinstance(body, DatasetCreateRequest)
        id = body.name + "_id"
        if body.name == 'xyz-no-tags':
            id = 'xyz-no-tags'
        dataset = DatasetData(id=id, name=body.name, last_modified_at=len(self.datasets) + 1,
                              type="", size_in_bytes=-1, n_samples=-1, created_at=-1)
        self.datasets += [dataset]
        response_ = CreateEntityResponse(id=id)
        return response_

    def get_dataset_by_id(self, dataset_id):
        _check_dataset_id(dataset_id)
        return next(dataset for dataset in self.default_datasets if dataset_id == dataset.id)

    def register_dataset_upload_by_id(self, body, dataset_id):
        _check_dataset_id(dataset_id)
        return True

    def delete_dataset_by_id(self, dataset_id, **kwargs):
        _check_dataset_id(dataset_id)
        datasets_without_that_id = [dataset for dataset in self.datasets if dataset.id != dataset_id]
        assert len(datasets_without_that_id) == len(self.datasets) - 1
        self.datasets = datasets_without_that_id

class MockedVersioningApi(VersioningApi):
    def get_latest_pip_version(self, **kwargs):
        return "1.0.8"

    def get_minimum_compatible_pip_version(self, **kwargs):
        return "1.0.0"

class MockedQuotaApi(QuotaApi):
    def get_quota_maximum_dataset_size(self, **kwargs):
        return "60000"

def mocked_request_put(dst_url: str, data=IOBase) -> Response:
    assert isinstance(dst_url, str)
    content_bytes: bytes = data.read()
    content_str: str = content_bytes.decode('utf-8')
    assert content_str.startswith('filenames')
    response_ = Response()
    response_.status_code = 200
    return response_


class MockedApiClient(ApiClient):
    def request(self, method, url, query_params=None, headers=None,
                post_params=None, body=None, _preload_content=True,
                _request_timeout=None):
        raise ValueError("ERROR: calling ApiClient.request(), but this should be mocked.")

    def call_api(self, resource_path, method,
                 path_params=None, query_params=None, header_params=None,
                 body=None, post_params=None, files=None,
                 response_type=None, auth_settings=None, async_req=None,
                 _return_http_data_only=None, collection_formats=None,
                 _preload_content=True, _request_timeout=None):
        raise ValueError("ERROR: calling ApiClient.call_api(), but this should be mocked.")


class MockedApiWorkflowClient(ApiWorkflowClient):

    embeddings_filename_base = 'img'
    n_embedding_rows_on_server = N_FILES_ON_SERVER

    def __init__(self, *args, **kwargs):
        lightly.api.api_workflow_client.ApiClient = MockedApiClient
        lightly.api.version_checking.VersioningApi = MockedVersioningApi
        ApiWorkflowClient.__init__(self, *args, **kwargs)

        self._samplings_api = MockedSamplingsApi(api_client=self.api_client)
        self._jobs_api = MockedJobsApi(api_client=self.api_client)
        self._tags_api = MockedTagsApi(api_client=self.api_client)
        self._embeddings_api = MockedEmbeddingsApi(api_client=self.api_client)
        self._samples_api = MockedSamplesApi(api_client=self.api_client)
        self._mappings_api = MockedMappingsApi(api_client=self.api_client,
                                              samples_api=self._samples_api)
        self._scores_api = MockedScoresApi(api_client=self.api_client)
        self._datasets_api = MockedDatasetsApi(api_client=self.api_client)
        self._quota_api = MockedQuotaApi(api_client=self.api_client)

        lightly.api.api_workflow_client.requests.put = mocked_request_put

        self.wait_time_till_next_poll = 0.001  # for api_workflow_sampling

    def upload_file_with_signed_url(
            self, file: IOBase, signed_write_url: str,
            max_backoff: int = 32, max_retries: int = 5
    ) -> Response:
        res = Response()
        return res

    def _get_csv_reader_from_read_url(self, read_url: str):
        n_rows: int = self.n_embedding_rows_on_server
        n_dims: int = self.n_dims_embeddings_on_server

        rows_csv = [['filenames'] + [f'embeddings_{i}' for i in range(n_dims)] + ['labels']]
        for i in range(n_rows):
            row = [f'{self.embeddings_filename_base}_{i}.jpg']
            for _ in range(n_dims):
                row.append(np.random.uniform(0, 1))
            row.append(i)
            rows_csv.append(row)

        # save the csv rows in a temporary in-memory string file
        # using a csv writer and then read them as bytes
        f = tempfile.SpooledTemporaryFile(mode="rw")
        writer = csv.writer(f)
        writer.writerows(rows_csv)
        f.seek(0)
        buffer = io.StringIO(f.read())
        reader = csv.reader(buffer)

        return reader


class MockedApiWorkflowSetup(unittest.TestCase):
    EMBEDDINGS_FILENAME_BASE: str = 'sample'

    def setUp(self, token="token_xyz",  dataset_id="dataset_id_xyz") -> None:
        self.api_workflow_client = MockedApiWorkflowClient(token=token, dataset_id=dataset_id)
