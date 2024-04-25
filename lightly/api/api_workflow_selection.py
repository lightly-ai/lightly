import time
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from lightly.openapi_generated.swagger_client.models import (
    ActiveLearningScoreCreateRequest,
    JobState,
    JobStatusData,
    SamplingConfig,
    SamplingConfigStoppingCondition,
    SamplingCreateRequest,
    TagData,
)


def _parse_active_learning_scores(scores: Union[np.ndarray, List]):
    """Makes list/np.array of active learning scores serializable."""
    # the api only accepts float64s
    if isinstance(scores, np.ndarray):
        scores = scores.astype(np.float64)

    # convert to list and return
    return list(scores)


class _SelectionMixin:
    def upload_scores(
        self, al_scores: Dict[str, NDArray[np.float_]], query_tag_id: str
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
