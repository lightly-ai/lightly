from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

from lightly.active_learning.config.selection_config import SelectionConfig
from lightly.openapi_generated.swagger_client.models import (
    ActiveLearningScoreCreateRequest,
    JobState,
    JobStatusData,
    SamplingConfig,
    SamplingConfigStoppingCondition,
    SamplingCreateRequest,
    TagData,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _parse_active_learning_scores(scores: Union[np.ndarray, List]):
    """Makes list/np.array of active learning scores serializable."""
    # the api only accepts float64s
    if isinstance(scores, np.ndarray):
        scores = scores.astype(np.float64)

    # convert to list and return
    return list(scores)


class _SelectionMixin:
    def upload_scores(
        self, al_scores: Dict[str, NDArray[np.float64]], query_tag_id: str
    ) -> None:
        """Uploads active learning scores for a tag.

        Args:
            al_scores:
                Active learning scores. Must be a mapping between score names
                and score arrays. The length of each score array must match samples
                in the designated tag.
            query_tag_id: ID of the desired tag.

        :meta private:  # Skip docstring generation
        """
        # iterate over all available score types and upload them
        for score_type, score_values in al_scores.items():
            body = ActiveLearningScoreCreateRequest(
                score_type=score_type,
                scores=_parse_active_learning_scores(score_values),
            )
            self._scores_api.create_or_update_active_learning_score_by_tag_id(
                active_learning_score_create_request=body,
                dataset_id=self.dataset_id,
                tag_id=query_tag_id,
            )

    def selection(
        self,
        selection_config: SelectionConfig,
        preselected_tag_id: Optional[str] = None,
        query_tag_id: Optional[str] = None,
    ) -> TagData:
        """Performs a selection given the arguments.

        Args:
            selection_config:
                The configuration of the selection.
            preselected_tag_id:
                The tag defining the already chosen samples (e.g., already
                labelled ones). Optional.
            query_tag_id:
                ID of the tag where samples should be fetched. None resolves to
                `initial-tag`. Defaults to None.

        Returns:
            The newly created tag of the selection.

        Raises:
            RuntimeError:
                When a tag with the tag name specified in the selection config already exists.
                When `initial-tag` does not exist in the dataset.
                When the selection task fails.

        :meta private:  # Skip docstring generation
        """

        warnings.warn(
            DeprecationWarning(
                "ApiWorkflowClient.selection() is deprecated "
                "and will be removed in the future."
            ),
        )

        # make sure the tag name does not exist yet
        tags = self.get_all_tags()
        if selection_config.name in [tag.name for tag in tags]:
            raise RuntimeError(
                f"There already exists a tag with tag_name {selection_config.name}."
            )
        if len(tags) == 0:
            raise RuntimeError("There exists no initial-tag for this dataset.")

        # make sure we have an embedding id
        try:
            self.embedding_id
        except AttributeError:
            self.set_embedding_id_to_latest()

        # trigger the selection
        payload = self._create_selection_create_request(
            selection_config, preselected_tag_id, query_tag_id
        )
        payload.row_count = self.get_all_tags()[0].tot_size
        response = self._selection_api.trigger_sampling_by_id(
            sampling_create_request=payload,
            dataset_id=self.dataset_id,
            embedding_id=self.embedding_id,
        )
        job_id = response.job_id

        # poll the job status till the job is not running anymore
        exception_counter = 0  # TODO; remove after solving https://github.com/lightly-ai/lightly-core/issues/156
        job_status_data = None

        wait_time_till_next_poll = getattr(self, "wait_time_till_next_poll", 1)
        while (
            job_status_data is None
            or job_status_data.status == JobState.RUNNING
            or job_status_data.status == JobState.WAITING
            or job_status_data.status == JobState.UNKNOWN
        ):
            # sleep before polling again
            time.sleep(wait_time_till_next_poll)
            # try to read the sleep time until the next poll from the status data
            try:
                job_status_data: JobStatusData = self._jobs_api.get_job_status_by_id(
                    job_id=job_id
                )
                wait_time_till_next_poll = job_status_data.wait_time_till_next_poll
            except Exception as err:
                exception_counter += 1
                if exception_counter == 20:
                    print(
                        f"Selection job with job_id {job_id} could not be started because of error: {err}"
                    )
                    raise err

        if job_status_data.status == JobState.FAILED:
            raise RuntimeError(
                f"Selection job with job_id {job_id} failed with error {job_status_data.error}"
            )

        # get the new tag from the job status
        new_tag_id = job_status_data.result.data
        if new_tag_id is None:
            raise RuntimeError(f"TagId returned by job with job_id {job_id} is None.")
        new_tag_data = self._tags_api.get_tag_by_tag_id(
            dataset_id=self.dataset_id, tag_id=new_tag_id
        )

        return new_tag_data

    def _create_selection_create_request(
        self,
        selection_config: SelectionConfig,
        preselected_tag_id: Optional[str],
        query_tag_id: Optional[str],
    ) -> SamplingCreateRequest:
        """Creates a SamplingCreateRequest

        First, it checks how many samples are already labeled by
            getting the number of samples in the preselected_tag_id.
        Then the stopping_condition.n_samples
            is set to be the number of already labeled samples + the selection_config.batch_size.
        Last the SamplingCreateRequest is created with the necessary nested class instances.

        """

        sampling_config = SamplingConfig(
            stopping_condition=SamplingConfigStoppingCondition(
                n_samples=selection_config.n_samples,
                min_distance=selection_config.min_distance,
            )
        )
        sampling_create_request = SamplingCreateRequest(
            new_tag_name=selection_config.name,
            method=selection_config.method,
            config=sampling_config,
            preselected_tag_id=preselected_tag_id,
            query_tag_id=query_tag_id,
        )
        return sampling_create_request
