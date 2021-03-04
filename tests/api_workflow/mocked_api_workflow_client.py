import unittest

import lightly

from lightly.api.api_workflow_client import ApiWorkflowClient

from io import BufferedReader
from typing import *

from lightly.openapi_generated.swagger_client import ScoresApi, CreateEntityResponse, SamplesApi, SampleCreateRequest, \
    InitialTagCreateRequest, ApiClient
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
    def get_embeddings_csv_write_url_by_id(self, dataset_id: str, **kwargs):
        assert isinstance(dataset_id, str)
        response_ = WriteCSVUrlData(signed_write_url="signed_write_url_valid", embedding_id="embedding_id_xyz")
        return response_

    def get_embeddings_by_dataset_id(self, dataset_id, **kwargs) -> List[DatasetEmbeddingData]:
        assert isinstance(dataset_id, str)
        return [DatasetEmbeddingData(id="embedding_id_xyz", name="embedding_name_xxyyzz",
                                     is_processed=True, created_at=0)]


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


def mocked_upload_file_with_signed_url(file: str, url: str, mocked_return_value=True) -> bool:
    assert isinstance(file, BufferedReader)
    assert isinstance(url, str)
    return mocked_return_value


def mocked_get_quota(token: str) -> Tuple[int, int]:
    quota = 25000
    status = 200
    return quota, status


def mocked_put_request(dst_url, data=None, params=None, json=None, max_backoff=32, max_retries=5) -> bool:
    assert isinstance(dst_url, str)
    success = True
    return success


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
        ApiWorkflowClient.__init__(self, *args, **kwargs)

        self.samplings_api = MockedSamplingsApi(api_client=self.api_client)
        self.jobs_api = MockedJobsApi(api_client=self.api_client)
        self.tags_api = MockedTagsApi(api_client=self.api_client)
        self.embeddings_api = MockedEmbeddingsApi(api_client=self.api_client)
        self.mappings_api = MockedMappingsApi(api_client=self.api_client)
        self.scores_api = MockedScoresApi(api_client=self.api_client)
        self.samples_api = MockedSamplesApi(api_client=self.api_client)

        lightly.api.api_workflow_upload_dataset.get_quota = mocked_get_quota
        lightly.api.api_workflow_client.put_request = mocked_put_request

        self.wait_time_till_next_poll = 0.001  # for api_workflow_sampling


class MockedApiWorkflowSetup(unittest.TestCase):
    def setUp(self) -> None:
        self.api_workflow_client = MockedApiWorkflowClient(token="token_xyz", dataset_id="dataset_id_xyz")
