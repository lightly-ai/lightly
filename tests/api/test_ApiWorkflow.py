import unittest
from unittest.mock import patch
import tempfile
import os

import numpy as np

import lightly
from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.api.api_workflow import ApiWorkflow
from lightly.openapi_generated.swagger_client import EmbeddingsApi, SamplingsApi, TagsApi, JobsApi, JobStatusData, \
    SamplingCreateRequest, JobState, TagData, JobStatusDataResult, JobResultType
from lightly.openapi_generated.swagger_client.models.inline_response2002 import InlineResponse2002
from lightly.openapi_generated.swagger_client.models.inline_response2003 import InlineResponse2003


class MockedEmbeddingsApi(EmbeddingsApi):
    def get_embeddings_csv_write_url_by_id(self, dataset_id: str, **kwargs):
        assert isinstance(dataset_id, str)
        response = InlineResponse2002(signed_write_url="signed_write_url_valid", embedding_id="embedding_id_xyz")
        return response


class MockedSamplingsApi(SamplingsApi):
    def trigger_sampling_by_id(self, body, dataset_id, embedding_id, **kwargs):
        assert isinstance(body, SamplingCreateRequest)
        assert isinstance(dataset_id, str)
        assert isinstance(embedding_id, str)
        response = InlineResponse2003(job_id="155")
        return response


class MockedJobsApi(JobsApi):
    def __init__(self, *args, **kwargs):
        self.no_calls = 0
        JobsApi.__init__(self, *args, **kwargs)

    def get_job_status_by_id(self, job_id, **kwargs):
        assert isinstance(job_id, str)
        self.no_calls += 1
        status = JobState.FINISHED if self.no_calls > 3 else JobState.RUNNING
        result = JobStatusDataResult(type=JobResultType.SAMPLING, data="tag_id_xyz")
        response = JobStatusData(id="id_", status=status, wait_time_till_next_poll=0.5, created_at=1234, result=result)
        return response


class MockedTagsApi(TagsApi):
    def get_tag_by_tag_id(self, dataset_id, tag_id, **kwargs):
        assert isinstance(dataset_id, str)
        assert isinstance(tag_id, str)
        response = TagData(id=tag_id, dataset_id=dataset_id, prev_tag="initial-tag", bit_mask_data="0x80bda23e9",
                           name='second-tag', tot_size=0, created_at=1577836800, changes=dict())
        return response


def mocked_upload_file_with_signed_url(file: str, url: str, mocked_return_value=True) -> bool:
    assert isinstance(file, str)
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
        lightly.api.api_workflow.upload_file_with_signed_url = mocked_upload_file_with_signed_url
        api_workflow = ApiWorkflow(host="host_xyz", token="token_xyz", dataset_id="dataset_id_xyz")

        # perform the workflow to upload the embeddings
        api_workflow.upload_embeddings(path_to_embeddings_csv=path_to_embeddings)

    def test_sampling(self):
        lightly.api.api_workflow.SamplingsApi = MockedSamplingsApi
        lightly.api.api_workflow.TagsApi = MockedTagsApi
        lightly.api.api_workflow.JobsApi = MockedJobsApi
        api_workflow = ApiWorkflow(host="host_xyz", token="token_xyz", dataset_id="dataset_id_xyz")

        sampler_config = SamplerConfig()

        new_tag_data: TagData = api_workflow.sampling(sampler_config=sampler_config)
        assert isinstance(new_tag_data, TagData)
