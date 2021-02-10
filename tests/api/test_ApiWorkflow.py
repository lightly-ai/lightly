from io import BufferedReader
import tempfile
import os
from typing import *

import unittest
import numpy as np

import lightly
from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.api.api_workflow import ApiWorkflow
from lightly.openapi_generated.swagger_client import EmbeddingsApi, SamplingsApi, TagsApi, JobsApi, JobStatusData, \
    SamplingCreateRequest, JobState, TagData, JobStatusDataResult, JobResultType, MappingsApi, AsyncTaskData, \
    DatasetEmbeddingData
from lightly.openapi_generated.swagger_client.models.inline_response200 import InlineResponse200
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


class TestApiWorkflow(unittest.TestCase):

    def test_upload_embedding(self, n_data: int = 100):
        # create fake embeddings
        folder_path = tempfile.mkdtemp()
        path_to_embeddings = os.path.join(
            folder_path,
            'embeddings.csv'
        )
        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        labels = [0] * len(sample_names)
        lightly.utils.save_embeddings(
            path_to_embeddings,
            np.random.randn(n_data, 16),
            labels,
            sample_names
        )

        # Set the workflow with mocked functions
        lightly.api.api_workflow.EmbeddingsApi = MockedEmbeddingsApi
        lightly.api.api_workflow.MappingsApi = MockedMappingsApi
        lightly.api.api_workflow.upload_file_with_signed_url = mocked_upload_file_with_signed_url
        api_workflow = ApiWorkflow(host="host_xyz", token="token_xyz", dataset_id="dataset_id_xyz")

        # perform the workflow to upload the embeddings
        api_workflow.upload_embeddings(path_to_embeddings_csv=path_to_embeddings)

    def test_sampling(self):
        lightly.api.api_workflow.SamplingsApi = MockedSamplingsApi
        lightly.api.api_workflow.TagsApi = MockedTagsApi
        lightly.api.api_workflow.JobsApi = MockedJobsApi
        api_workflow = ApiWorkflow(host="host_xyz", token="token_xyz", dataset_id="dataset_id_xyz")
        api_workflow.embedding_id = "embedding_id_xyz"

        sampler_config = SamplerConfig()

        new_tag_data: TagData = api_workflow.sampling(sampler_config=sampler_config)
        assert isinstance(new_tag_data, TagData)
