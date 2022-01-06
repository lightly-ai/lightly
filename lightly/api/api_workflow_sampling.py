import time
from typing import Dict, List, Union

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


def _parse_active_learning_scores(scores: Union[np.ndarray, List]):
    """Makes list/np.array of active learning scores serializable.

    """
    # the api only accepts float64s
    if isinstance(scores, np.ndarray):
        scores = scores.astype(np.float64)

    # convert to list and return
    return list(scores)


class _SamplingMixin:

    def upload_scores(self, al_scores: Dict[str, np.ndarray], query_tag_id: str = None):

        tags = self.get_all_tags()

        # upload the active learning scores to the api
        # change @20210422: we store the active learning scores with the query
        # tag. policy is that if there's no explicit query tag, the whole dataset
        # will be the query tag (i.e. query_tag = initial-tag)
        # set the query tag to the initial-tag if necessary
        if query_tag_id is None:
            query_tag = next(t for t in tags if t.name == 'initial-tag')
            query_tag_id = query_tag.id
        # iterate over all available score types and upload them
        for score_type, score_values in al_scores.items():
            body = ActiveLearningScoreCreateRequest(
                score_type=score_type,
                scores=_parse_active_learning_scores(score_values)
            )
            self._scores_api.create_or_update_active_learning_score_by_tag_id(
                body,
                dataset_id=self.dataset_id,
                tag_id=query_tag_id,
            )

    def sampling(self, sampler_config: SamplerConfig, preselected_tag_id: str = None, query_tag_id: str = None) \
            -> TagData:
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
            ApiException
            ValueError
            RuntimeError

        """

        # make sure the tag name does not exist yet
        tags = self.get_all_tags()
        if sampler_config.name in [tag.name for tag in tags]:
            raise RuntimeError(f'There already exists a tag with tag_name {sampler_config.name}.')
        if len(tags) == 0:
            raise RuntimeError('There exists no initial-tag for this dataset.')

        # make sure we have an embedding id
        try:
            self.embedding_id
        except AttributeError:
            self.set_embedding_id_to_latest()

        # trigger the sampling
        payload = self._create_sampling_create_request(sampler_config, preselected_tag_id, query_tag_id)
        payload.row_count = self.get_all_tags()[0].tot_size
        response = self._samplings_api.trigger_sampling_by_id(payload, self.dataset_id, self.embedding_id)
        job_id = response.job_id

        # poll the job status till the job is not running anymore
        exception_counter = 0  # TODO; remove after solving https://github.com/lightly-ai/lightly-core/issues/156
        job_status_data = None

        wait_time_till_next_poll = getattr(self, "wait_time_till_next_poll", 1)
        while job_status_data is None \
                or job_status_data.status == JobState.RUNNING \
                or job_status_data.status == JobState.WAITING \
                or job_status_data.status == JobState.UNKNOWN:
            # sleep before polling again
            time.sleep(wait_time_till_next_poll)
            # try to read the sleep time until the next poll from the status data
            try:
                job_status_data: JobStatusData = self._jobs_api.get_job_status_by_id(job_id=job_id)
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
        if new_tag_id is None:
            raise RuntimeError(f"TagId returned by job with job_id {job_id} is None.")
        new_tag_data = self._tags_api.get_tag_by_tag_id(self.dataset_id, tag_id=new_tag_id)

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
