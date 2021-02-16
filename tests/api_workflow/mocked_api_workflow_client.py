import lightly

from lightly.api.api_workflow_client import ApiWorkflowClient

from io import BufferedReader
from typing import *

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
        assert isinstance(dataset_id,str)
        return [DatasetEmbeddingData(id="embedding_id_xyz", name="embedding_name_xxyyzz",
                                     is_processed=True, created_at=0)]


class MockedSamplingsApi(SamplingsApi):
    def trigger_sampling_by_id(self, body, dataset_id, embedding_id, **kwargs):
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
            result = JobStatusDataResult(type=JobResultType.SAMPLING, data="tag_id_xyz")
            response_ = JobStatusData(id="id_", status=JobState.FINISHED, wait_time_till_next_poll=0,
                                      created_at=1234, finished_at=1357, result=result)
        else:
            result = None
            response_ = JobStatusData(id="id_", status=JobState.RUNNING, wait_time_till_next_poll=0.5,
                                      created_at=1234, result=result)
        return response_


class MockedTagsApi(TagsApi):
    def get_tag_by_tag_id(self, dataset_id, tag_id, **kwargs):
        assert isinstance(dataset_id, str)
        assert isinstance(tag_id, str)
        response_ = TagData(id=tag_id, dataset_id=dataset_id, prev_tag_id="initial-tag", bit_mask_data="0x80bda23e9",
                            name='second-tag', tot_size=0, created_at=1577836800, changes=dict())
        return response_


class MockedMappingsApi(MappingsApi):
    def __init__(self, *args, **kwargs):
        sample_names = [f'img_{i}.jpg' for i in range(100)]
        sample_names.reverse()
        self.sample_names = sample_names
        MappingsApi.__init__(self, *args, **kwargs)

    def get_sample_mappings_by_dataset_id(self, dataset_id, field, **kwargs):
        return self.sample_names


def mocked_upload_file_with_signed_url(file: str, url: str, mocked_return_value=True) -> bool:
    assert isinstance(file, BufferedReader)
    assert isinstance(url, str)
    return mocked_return_value


class MockedApiWorkflowClient(ApiWorkflowClient):
    def __init__(self, *args, **kwargs):
        ApiWorkflowClient.__init__(self, *args, **kwargs)

        self.samplings_api = MockedSamplingsApi(api_client=self.api_client)
        self.jobs_api = MockedJobsApi(api_client=self.api_client)
        self.tags_api = MockedTagsApi(api_client=self.api_client)
        self.embeddings_api = MockedEmbeddingsApi(api_client=self.api_client)
        self.mappings_api = MockedMappingsApi(api_client=self.api_client)
        lightly.api_client.api_workflow_upload_embeddings.upload_file_with_signed_url = mocked_upload_file_with_signed_url


