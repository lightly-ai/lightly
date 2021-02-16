from typing import *

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.scorer import Scorer
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.api.bitmask import BitMask


class ActiveLearningAgent:
    def __init__(self, api_workflow_client: ApiWorkflowClient):

        self.api_workflow_client = api_workflow_client

    def sample(self, sampler_config: SamplerConfig, al_scorer: Scorer = None, preselected_tag_id: str = None) \
            -> Tuple[List[int], List[str]]:

        # calculate scores
        if al_scorer is not None:
            scores = al_scorer._calculate_scores()
        else:
            scores = None

        # perform the sampling
        new_tag_data = self.api_workflow_client.sampling(
            sampler_config=sampler_config, al_scores=scores, preselected_tag_id=preselected_tag_id)

        # extract the chosen samples
        chosen_samples_ids = BitMask.from_hex(new_tag_data.bit_mask_data).to_indices()
        chosen_filenames = [self.api_workflow_client.filenames_on_server[i] for i in chosen_samples_ids]

        return chosen_samples_ids, chosen_filenames
