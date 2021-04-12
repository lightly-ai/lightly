import unittest
from io import IOBase

from requests import Response

from lightly.openapi_generated.swagger_client.models.dataset_create_request import DatasetCreateRequest

from lightly.openapi_generated.swagger_client.models.dataset_data import DatasetData

from lightly.openapi_generated.swagger_client.api.datasets_api import DatasetsApi

import lightly

from lightly.api.api_workflow_client import ApiWorkflowClient

from typing import *

from lightly.openapi_generated.swagger_client import ScoresApi, CreateEntityResponse, SamplesApi, SampleCreateRequest, \
    InitialTagCreateRequest, ApiClient, VersioningApi, QuotaApi
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


class MockedEmbeddingsApi(EmbeddingsApi):
    def __init__(self, api_client):
        EmbeddingsApi.__init__(self, api_client=api_client)
        self.embeddings = [DatasetEmbeddingData(id="embedding_id_xyz", name="embedding_name_xxyyzz",
                                                is_processed=True, created_at=0)]

    def get_embeddings_csv_write_url_by_id(self, dataset_id: str, **kwargs):
        assert isinstance(dataset_id, str)
        response_ = WriteCSVUrlData(signed_write_url="signed_write_url_valid", embedding_id="embedding_id_xyz")
        return response_

    def get_embeddings_by_dataset_id(self, dataset_id, **kwargs) -> List[DatasetEmbeddingData]:
        assert isinstance(dataset_id, str)
        return self.embeddings


class MockedSamplingsApi(SamplingsApi):
    def trigger_sampling_by_id(self, body: SamplingCreateRequest, dataset_id, embedding_id, **kwargs):
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
        assert isinstance(body, InitialTagCreateRequest)
        assert isinstance(dataset_id, str)
        response_ = CreateEntityResponse(id="xyz")
        return response_

    def get_tag_by_tag_id(self, dataset_id, tag_id, **kwargs):
        assert isinstance(dataset_id, str)
        assert isinstance(tag_id, str)
        response_ = TagData(id=tag_id, dataset_id=dataset_id, prev_tag_id="initial-tag", bit_mask_data="0x80bda23e9",
                            name='second-tag', tot_size=15, created_at=1577836800, changes=dict())
        return response_

    def get_tags_by_dataset_id(self, dataset_id, **kwargs):
        tag_1 = TagData(id='inital_tag_id', dataset_id=dataset_id, prev_tag_id=None,
                        bit_mask_data="0x80bda23e9", name='initial-tag', tot_size=15,
                        created_at=1577836800, changes=dict())
        tag_2 = TagData(id='query_tag_id_xyz', dataset_id=dataset_id, prev_tag_id="initial-tag",
                        bit_mask_data="0x80bda23e9", name='query_tag_name_xyz', tot_size=15,
                        created_at=1577836800, changes=dict())
        tag_3 = TagData(id='preselected_tag_id_xyz', dataset_id=dataset_id, prev_tag_id="initial-tag",
                        bit_mask_data="0x80bda23e9", name='preselected_tag_name_xyz', tot_size=15,
                        created_at=1577836800, changes=dict())
        tags = [tag_1, tag_2, tag_3]
        no_tags_to_return = getattr(self, "no_tags", 3)
        tags = tags[:no_tags_to_return]
        return tags


class MockedScoresApi(ScoresApi):
    def create_or_update_active_learning_score_by_tag_id(self, body, dataset_id, tag_id, **kwargs) -> \
            CreateEntityResponse:
        if len(body.scores) > 0 and not isinstance(body.scores[0], float):
            raise AttributeError
        response_ = CreateEntityResponse(id="sampled_tag_id_xyz")
        return response_


class MockedMappingsApi(MappingsApi):
    def __init__(self, *args, **kwargs):
        sample_names = [f'img_{i}.jpg' for i in range(100)]
        sample_names.reverse()
        self.sample_names = sample_names
        MappingsApi.__init__(self, *args, **kwargs)

    def get_sample_mappings_by_dataset_id(self, dataset_id, field, **kwargs):
        return self.sample_names


class MockedSamplesApi(SamplesApi):
    def create_sample_by_dataset_id(self, body, dataset_id, **kwargs):
        assert isinstance(body, SampleCreateRequest)
        response_ = CreateEntityResponse(id="xyz")
        return response_

    def get_sample_image_write_url_by_id(self, dataset_id, sample_id, is_thumbnail, **kwargs):
        url = f"{sample_id}_write_url"
        return url

    def get_sample_image_read_url_by_id(self, dataset_id, sample_id, type, **kwargs):
        url = f"{sample_id}_write_url"
        return url


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
        dataset = DatasetData(id=id, name=body.name, last_modified_at=len(self.datasets) + 1,
                              type="", size_in_bytes=-1, n_samples=-1, created_at=-1)
        self.datasets += [dataset]
        response_ = CreateEntityResponse(id=id)
        return response_

    def get_dataset_by_id(self, dataset_id):
        return next(dataset for dataset in self.default_datasets if dataset_id == dataset.id)

    def delete_dataset_by_id(self, dataset_id, **kwargs):
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
    assert isinstance(data, IOBase)
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
    def __init__(self, *args, **kwargs):
        lightly.api.api_workflow_client.ApiClient = MockedApiClient
        lightly.api.version_checking.VersioningApi = MockedVersioningApi
        ApiWorkflowClient.__init__(self, *args, **kwargs)

        self.samplings_api = MockedSamplingsApi(api_client=self.api_client)
        self.jobs_api = MockedJobsApi(api_client=self.api_client)
        self.tags_api = MockedTagsApi(api_client=self.api_client)
        self.embeddings_api = MockedEmbeddingsApi(api_client=self.api_client)
        self.mappings_api = MockedMappingsApi(api_client=self.api_client)
        self.scores_api = MockedScoresApi(api_client=self.api_client)
        self.samples_api = MockedSamplesApi(api_client=self.api_client)
        self.datasets_api = MockedDatasetsApi(api_client=self.api_client)
        self.quota_api = MockedQuotaApi(api_client=self.api_client)

        lightly.api.api_workflow_client.requests.put = mocked_request_put

        self.wait_time_till_next_poll = 0.001  # for api_workflow_sampling


class MockedApiWorkflowSetup(unittest.TestCase):
    def setUp(self, token="token_xyz",  dataset_id="dataset_id_xyz") -> None:
        self.api_workflow_client = MockedApiWorkflowClient(token=token, dataset_id=dataset_id)
