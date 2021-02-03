import time
from typing import *

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.openapi_generated.swagger_client import Configuration, ApiClient, SamplingsApi, JobsApi, JobState, \
    TagsApi, JobStatusData, EmbeddingsApi
from lightly.api.upload import upload_file_with_signed_url
from lightly.openapi_generated.swagger_client.models.inline_response2002 import InlineResponse2002


class ApiWorkflow:
    def __init__(self, host: str, token: str, dataset_id: str, embedding_id: str = None):

        configuration = Configuration()
        configuration.host = host
        configuration.api_key = {'token': token}
        api_client = ApiClient(configuration=configuration)
        self.api_client = api_client

        self.dataset_id = dataset_id
        if embedding_id is not None:
            self.embedding_id = embedding_id

        self.samplings_api = SamplingsApi(api_client=self.api_client)
        self.jobs_api = JobsApi(api_client=self.api_client)
        self.tags_api = TagsApi(api_client=self.api_client)
        self.embeddings_api = EmbeddingsApi(api_client=api_client)

    def sampling(self, sampler_config: SamplerConfig, preselected_tag_id: str = None, query_tag_id: str = None,
                 al_scores: Dict[str, List[int]] = None):

        # upload the active learning scores to the api
        if al_scores is not None:
            raise NotImplementedError

        # trigger the sampling
        payload = sampler_config.get_as_api_sampling_create_request(
            preselected_tag_id=preselected_tag_id, query_tag_id=query_tag_id)
        response = self.samplings_api.trigger_sampling_by_id(payload, self.dataset_id, "embedding_id_xyz")
        job_id = response.job_id

        # poll the job status till the job is finished
        time.sleep(2)
        while True:
            job_status_data: JobStatusData = self.jobs_api.get_job_status_by_id(job_id=job_id)
            if job_status_data.status == JobState.FINISHED:
                new_tag = job_status_data.result.data
                break
            elif job_status_data.status == JobState.FAILED:
                raise ValueError(f"Sampling job failed with error {job_status_data.error}")

            time.sleep(job_status_data.wait_time_till_next_poll)

        # get the new tag from the job status
        new_tag_data = self.tags_api.get_tag_by_tag_id(self.dataset_id, tag_id=new_tag)

        return new_tag_data

    def upload_embeddings(self, path_to_embeddings_csv: str, name: str = None):

        # TODO: load the csv, sort it by the order given by the tag, save the csv in order

        response: InlineResponse2002 = \
            self.embeddings_api.get_embeddings_csv_write_url_by_id(self.dataset_id, name=name)
        self.embedding_id = response.embedding_id
        signed_write_url = response.signed_write_url

        upload_file_with_signed_url(file=path_to_embeddings_csv, url=signed_write_url)
