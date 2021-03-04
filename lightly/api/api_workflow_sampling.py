from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lightly.api.api_workflow_client import ApiWorkflowClient

import time
from typing import Dict, List

import numpy as np

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.openapi_generated.swagger_client import ActiveLearningScoreCreateRequest
from lightly.openapi_generated.swagger_client.models.job_state import JobState
from lightly.openapi_generated.swagger_client.models.job_status_data import JobStatusData
from lightly.openapi_generated.swagger_client.models.tag_data import TagData
from lightly.openapi_generated.swagger_client.models.sampling_config import SamplingConfig
from lightly.openapi_generated.swagger_client.models.sampling_create_request import SamplingCreateRequest
from lightly.openapi_generated.swagger_client.models.sampling_config_stopping_condition import \
    SamplingConfigStoppingCondition


class _SamplingMixin:

    def sampling(self: ApiWorkflowClient, sampler_config: SamplerConfig, al_scores: Dict[str, List[np.ndarray]] = None,
                 preselected_tag_id: str = None, query_tag_id: str = None) -> TagData:
        """Performs a sampling given the arguments.

        Args:
            sampler_config:
                The configuration of the sampler.
            al_scores:
                The active learning scores for the sampler.
            preselected_tag_id:
                The tag defining the already chosen samples (e.g. already labelled ones), default: None.
            query_tag_id:
                The tag defining where to sample from, default: None resolves to the initial-tag.

        Returns:
            The newly created tag of the sampling.

        Raises:
            ApiException, ValueError, RuntimeError

        """

        # make sure the tag name does not exist yet
        tags = self._get_all_tags()
        if sampler_config.name in [tag.name for tag in tags]:
            raise RuntimeError('There already exists a tag with tag_name {sampler_config.name}.')

        # make sure we have an embedding id
        try:
            self.embedding_id
        except AttributeError:
            self.set_embedding_id_by_name()


        # upload the active learning scores to the api
        if al_scores is not None:
            if preselected_tag_id is None:
                raise ValueError
            for score_type, score_values in al_scores.items():
                body = ActiveLearningScoreCreateRequest(score_type=score_type, scores=list(score_values))
                self.scores_api.create_or_update_active_learning_score_by_tag_id(
                    body, dataset_id=self.dataset_id, tag_id=preselected_tag_id)

        # trigger the sampling
        payload = self._create_sampling_create_request(sampler_config, preselected_tag_id, query_tag_id)
        payload.row_count = self._get_all_tags()[0].tot_size
        response = self.samplings_api.trigger_sampling_by_id(payload, self.dataset_id, self.embedding_id)
        job_id = response.job_id

        # poll the job status till the job is not running anymore
        exception_counter = 0  # TODO; remove after solving https://github.com/lightly-ai/lightly-core/issues/156
        job_status_data = None

        wait_time_till_next_poll = getattr(self, "wait_time_till_next_poll", 1)
        while job_status_data is None or job_status_data.status == JobState.RUNNING:
            time.sleep(wait_time_till_next_poll)
            try:
                job_status_data: JobStatusData = self.jobs_api.get_job_status_by_id(job_id=job_id)
                wait_time_till_next_poll = job_status_data.wait_time_till_next_poll
            except Exception as err:
                exception_counter += 1
                if exception_counter == 20:
                    print(f"Sampling job with job_id {job_id} could not be started because of error: {err}")
                    raise err

        if job_status_data.status == JobState.FAILED:
            raise RuntimeError(f"Sampling job with job_id {job_id} failed with error {job_status_data.error}")

        # get the new tag from the job status
        new_tag_id = job_status_data.result.data
        new_tag_data = self.tags_api.get_tag_by_tag_id(self.dataset_id, tag_id=new_tag_id)

        return new_tag_data

    def _create_sampling_create_request(self, sampler_config: SamplerConfig, preselected_tag_id: str, query_tag_id: str
                                        ) -> SamplingCreateRequest:
        """Creates a SamplingCreateRequest

        First, it checks how many samples are already labeled by
            getting the number of samples in the preselected_tag_id.
        Then the stopping_condition.n_samples
            is set to be the number of already labeled samples + the sampler_config.batch_size.
        Last the SamplingCreateRequest is created with the necessary nested class instances.

        """

        sampling_config = SamplingConfig(
            stopping_condition=SamplingConfigStoppingCondition(
                n_samples=sampler_config.n_samples,
                min_distance=sampler_config.min_distance
            )
        )
        sampling_create_request = SamplingCreateRequest(new_tag_name=sampler_config.name,
                                                        method=sampler_config.method,
                                                        config=sampling_config,
                                                        preselected_tag_id=preselected_tag_id,
                                                        query_tag_id=query_tag_id)
        return sampling_create_request

