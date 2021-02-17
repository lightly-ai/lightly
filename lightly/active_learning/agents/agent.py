from typing import *

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.scorer import Scorer
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.api.bitmask import BitMask


class ActiveLearningAgent:
    """A basic class providing an active learning policy

    Attributes:
        api_workflow_client:
            The client to connect to the api.
        preselected_tag_id:
            The id of the tag containing the already labeled samples, default: None == no labeled samples yet.
        query_tag_id:
            The id of the tag defining where to sample from, default: None resolves to initial_tag

    """

    def __init__(self, api_workflow_client: ApiWorkflowClient, query_tag_name: str = None, preselected_tag_name: str = None):

        self.api_workflow_client = api_workflow_client
        if query_tag_name is not None or preselected_tag_name is not None:
            tag_name_id_dict = dict([tag.name, tag.id] for tag in self.api_workflow_client._get_all_tags())
            if query_tag_name is not None:
                self.query_tag_id = tag_name_id_dict[query_tag_name]
            if preselected_tag_name is not None:
                self.preselected_tag_id_tag_id = tag_name_id_dict[preselected_tag_name]

        if not hasattr(self, "preselected_tag_id"):
            self.preselected_tag_id = None
        if not hasattr(self, "query_tag_id"):
            self.query_tag_id = None

    @property
    def labeled_set(self) -> List[str]:
        if self.preselected_tag_id is None:
            filenames = []
        else:
            tag_data = self.api_workflow_client.tags_api.get_tag_by_tag_id(
                self.api_workflow_client.dataset_id, tag_id=self.preselected_tag_id)
            chosen_samples_ids = BitMask.from_hex(tag_data.bit_mask_data).to_indices()
            filenames = [self.api_workflow_client.filenames_on_server[i] for i in chosen_samples_ids]
        return filenames

    @property
    def unlabeled_set(self) -> List[str]:
        if self.query_tag_id is None:
            filenames = self.api_workflow_client.filenames_on_server
        else:
            tag_data = self.api_workflow_client.tags_api.get_tag_by_tag_id(
                self.api_workflow_client.dataset_id, tag_id=self.query_tag_id)
            chosen_samples_ids = BitMask.from_hex(tag_data.bit_mask_data).to_indices()
            filenames = [self.api_workflow_client.filenames_on_server[i] for i in chosen_samples_ids]
        filenames_labeled = set(self.labeled_set)
        filenames = [f for f in filenames if f not in filenames_labeled]
        return filenames

    def query(self, sampler_config: SamplerConfig, al_scorer: Scorer = None) -> List[str]:
        """Performs an active learning query

        Args:
            sampler_config:
                The config of the sampler.
            al_scorer:
                An instance of a class inheriting from Scorer, e.g. a ClassificationScorer.

        Returns:
            the filenames of the chosen samples

        """
        # calculate scores
        if al_scorer is not None:
            scores = al_scorer._calculate_scores()
        else:
            scores = None

        # perform the sampling
        new_tag_data = self.api_workflow_client.sampling(
            sampler_config=sampler_config,
            al_scores=scores,
            preselected_tag_id=self.preselected_tag_id,
            query_tag_id=self.query_tag_id)

        # set the newly chosen tag as the new preselected_tag_id
        self.preselected_tag_id = new_tag_data.id

        return self.labeled_set
