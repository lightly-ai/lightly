from typing import *

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.scorer import Scorer
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.api.bitmask import BitMask
from lightly.openapi_generated.swagger_client.models import TagData


class ActiveLearningAgent:
    """A basic class providing an active learning policy

    Attributes:
        api_workflow_client:
            The client to connect to the api.
        preselected_tag_id:
            The id of the tag containing the already labeled samples, default: None == no labeled samples yet.
        query_tag_id:
            The id of the tag defining where to sample from, default: None resolves to initial_tag
        labeled_set:
            the filenames of the samples in the labeled set, List[str]
        unlabeled_set:
            the filenames of the samples in the unlabeled set, List[str]

    """

    def __init__(self, api_workflow_client: ApiWorkflowClient, query_tag_name: str = None, preselected_tag_name: str = None):

        self.api_workflow_client = api_workflow_client
        if query_tag_name is not None or preselected_tag_name is not None:
            tag_name_id_dict = dict([tag.name, tag.id] for tag in self.api_workflow_client._get_all_tags())
            if preselected_tag_name is not None:
                self.preselected_tag_id = tag_name_id_dict[preselected_tag_name]
            if query_tag_name is not None:
                self.query_tag_id = tag_name_id_dict[query_tag_name]

        if not hasattr(self, "preselected_tag_id"):
            self.preselected_tag_id = None
        if not hasattr(self, "query_tag_id"):
            self.query_tag_id = None
        self._set_labeled_and_unlabeled_set()

    def _set_labeled_and_unlabeled_set(self, preselected_tag_data: TagData = None):
        """Sets the labeled and unlabeled set based on the preselected and query tag id

        It loads the bitmaks for the both tag_ids from the server and then
        extracts the filenames from it given the mapping on the server.

        Args:
            preselected_tag_data:
                optional param, then it must not be loaded from the API

        """
        if self.preselected_tag_id is None:
            self.labeled_set = []
        else:
            if preselected_tag_data is None:
                preselected_tag_data = self.api_workflow_client.tags_api.get_tag_by_tag_id(
                    self.api_workflow_client.dataset_id, tag_id=self.preselected_tag_id)
            chosen_samples_ids = BitMask.from_hex(preselected_tag_data.bit_mask_data).to_indices()
            self.labeled_set = [self.api_workflow_client.filenames_on_server[i] for i in chosen_samples_ids]

        if not hasattr(self, "unlabeled_set"):
            if self.query_tag_id is None:
                self.unlabeled_set = self.api_workflow_client.filenames_on_server
            else:
                query_tag_data = self.api_workflow_client.tags_api.get_tag_by_tag_id(
                    self.api_workflow_client.dataset_id, tag_id=self.query_tag_id)
                chosen_samples_ids = BitMask.from_hex(query_tag_data.bit_mask_data).to_indices()
                self.unlabeled_set = [self.api_workflow_client.filenames_on_server[i] for i in chosen_samples_ids]

        filenames_labeled = set(self.labeled_set)
        self.unlabeled_set = [f for f in self.unlabeled_set if f not in filenames_labeled]

    def query(self, sampler_config: SamplerConfig, al_scorer: Scorer = None) -> List[str]:
        """Performs an active learning query

        As part of it, the self.labeled_set and self.unlabeled_set are updated and should be used for the next step.
        Args:
            sampler_config:
                The config of the sampler.
            al_scorer:
                An instance of a class inheriting from Scorer, e.g. a ClassificationScorer.

        Returns:
            the filenames of the samples in the new labeled_set

        """
        # check input
        if sampler_config.n_samples < len(self.labeled_set):
            print("ERROR: The number of samples which should be sampled according to the config"
                  " (including the current labeled set)"
                  "is smaller than the number of samples in the current labeled set.")
            return self.labeled_set

        # calculate scores
        if al_scorer is not None:
            no_unlabeled_samples = len(self.unlabeled_set)
            no_samples_with_predictions = len(al_scorer.model_output)
            if no_unlabeled_samples != no_samples_with_predictions:
                raise ValueError(f"The scorer must have exactly as much samples as in the unlabeled set,"
                                 f"but there are {no_samples_with_predictions} predictions in the scorer,"
                                 f"but {no_unlabeled_samples} in the unlabeled set.")
            scores_dict = al_scorer._calculate_scores()
        else:
            scores_dict = None

        # perform the sampling
        new_tag_data = self.api_workflow_client.sampling(
            sampler_config=sampler_config,
            al_scores=scores_dict,
            preselected_tag_id=self.preselected_tag_id,
            query_tag_id=self.query_tag_id)

        # set the newly chosen tag as the new preselected_tag_id and update the sets
        self.preselected_tag_id = new_tag_data.id
        self._set_labeled_and_unlabeled_set(new_tag_data)

        return self.labeled_set
