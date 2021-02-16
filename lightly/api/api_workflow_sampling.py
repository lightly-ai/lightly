import time
from typing import *

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.openapi_generated.swagger_client.models.job_state import JobState
from lightly.openapi_generated.swagger_client.models.job_status_data import JobStatusData
from lightly.openapi_generated.swagger_client.models.tag_data import TagData
from lightly.openapi_generated.swagger_client.rest import ApiException


class _SamplingMixin:
    def sampling(self, sampler_config: SamplerConfig, preselected_tag_id: str = None, query_tag_id: str = None,
                 al_scores: Dict[str, List[int]] = None) -> TagData:
        """Performs a sampling given the arguments.

        Args:
            sampler_config: The configuration of the sampler
            preselected_tag_id: The tag defining the already chosen samples (e.g. already labelled ones), default: None
            query_tag_id: The tag defining where to sample from, default: initial_tag
            al_scores: optional: the active learning scores for the sampler

        Returns:
            the newly created tag of the sampling

        Raises:
            ApiException, ValueError
        """

        # upload the active learning scores to the api
        if al_scores is not None:
            raise NotImplementedError  # TODO: fill out in later branches

        # trigger the sampling
        payload = sampler_config._get_as_api_sampling_create_request(
            preselected_tag_id=preselected_tag_id, query_tag_id=query_tag_id)
        payload.row_count = 15  # TODO: remove after solving https://github.com/lightly-ai/lightly-core/issues/150
        response = self.samplings_api.trigger_sampling_by_id(payload, self.dataset_id, self.embedding_id)
        job_id = response.job_id

        # poll the job status till the job is finished
        exception_counter = 0  # TODO; remove after solving https://github.com/lightly-ai/lightly-core/issues/150
        while True:
            try:
                job_status_data: JobStatusData = self.jobs_api.get_job_status_by_id(job_id=job_id)
            except Exception as e:
                exception_counter += 1
                if exception_counter == 10:
                    print(f"Sampling job with job_id {job_id} could not be started because of error: {e}")
                time.sleep(1)
                continue

            if job_status_data.status == JobState.FINISHED:
                break
            elif job_status_data.status == JobState.FAILED:
                raise ValueError(f"Sampling job with job_id {job_id} failed with error {job_status_data.error}")
            time.sleep(job_status_data.wait_time_till_next_poll)

        # get the new tag from the job status
        new_tag_id = job_status_data.result.data
        new_tag_data = self.tags_api.get_tag_by_tag_id(self.dataset_id, tag_id=new_tag_id)

        return new_tag_data
