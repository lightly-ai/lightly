from typing import *

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.scorer import Scorer
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.api.bitmask import BitMask


class ActiveLearningAgent:
    """A basic class providing an active learning policy

    Args:
        api_workflow_client: The client to connect to the api
        preselected_tag_id: The tag defining the already chosen samples (e.g. already labelled ones), default: None
    """
    def __init__(self, api_workflow_client: ApiWorkflowClient,
                 preselected_tag_id: str = None):

        self.preselected_tag_id = preselected_tag_id
        self.api_workflow_client = api_workflow_client

    def sample(self, sampler_config: SamplerConfig, al_scorer: Scorer = None, query_tag_id: str = None) \
            -> Tuple[List[int], List[str]]:
        """Performs an active learning sampling

        Args:
            sampler_config: the config of the sampler
            al_scorer: an instance of a class inheriting form Scorer, e.g. a ClassificationScorer
            query_tag_id: The tag defining where to sample from, default: initial_tag

        Returns:
            Tuple:
                the indexes of the chosen samples (indexes relative to all samples in the initial tag)
                the filenames of the chosen samples

        """
        # calculate scores
        if al_scorer is not None:
            scores = al_scorer._calculate_scores()
        else:
            scores = None

        # perform the sampling
        new_tag_data = self.api_workflow_client.sampling(
            sampler_config=sampler_config, al_scores=scores,
            preselected_tag_id=self.preselected_tag_id, query_tag_id=query_tag_id)

        # set the newly chosen tag as the new preselected_tag_id
        self.preselected_tag_id = new_tag_data.id

        # extract the chosen samples
        chosen_samples_ids = BitMask.from_hex(new_tag_data.bit_mask_data).to_indices()
        chosen_filenames = [self.api_workflow_client.filenames_on_server[i] for i in chosen_samples_ids]

        return chosen_samples_ids, chosen_filenames
