import time
from typing import *

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.openapi_generated.swagger_client.models.job_state import JobState
from lightly.openapi_generated.swagger_client.models.job_status_data import JobStatusData
from lightly.openapi_generated.swagger_client.models.tag_data import TagData


class _SamplingMixin:
    def sampling(self, sampler_config: SamplerConfig, preselected_tag_id: str = None, query_tag_id: str = None,
                 al_scores: Dict[str, List[int]] = None) -> TagData:
        # upload the active learning scores to the api
        if al_scores is not None:
            raise NotImplementedError

        # trigger the sampling
        payload = sampler_config.get_as_api_sampling_create_request(
            preselected_tag_id=preselected_tag_id, query_tag_id=query_tag_id)
        payload.row_count = 15  # TODO: remove after solving https://github.com/lightly-ai/lightly-core/issues/150
        response = self.samplings_api.trigger_sampling_by_id(payload, self.dataset_id, self.embedding_id)
        job_id = response.job_id
        print(f"job_id: {job_id}")

        # poll the job status till the job is finished
        while True:
            try:
                job_status_data: JobStatusData = self.jobs_api.get_job_status_by_id(job_id=job_id)
                if job_status_data.status == JobState.FINISHED:
                    new_tag = job_status_data.result.data
                    break
                elif job_status_data.status == JobState.FAILED:
                    raise ValueError(f"Sampling job failed with error {job_status_data.error}")
            except Exception as e:
                time.sleep(2)

            time.sleep(job_status_data.wait_time_till_next_poll)

        # get the new tag from the job status
        new_tag_data = self.tags_api.get_tag_by_tag_id(self.dataset_id, tag_id=new_tag)

        return new_tag_data
