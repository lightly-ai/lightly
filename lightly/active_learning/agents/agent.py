from typing import *
import warnings

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.scorer import Scorer
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.api.bitmask import BitMask
from lightly.openapi_generated.swagger_client.models import TagData


class ActiveLearningAgent:
    """Interface for active learning queries.

    Attributes:
        api_workflow_client:
            The client to connect to the api.
        preselected_tag_id:
            The id of the tag containing the already labeled samples, default: None == no labeled samples yet.
        query_tag_id:
            The id of the tag defining where to sample from, default: None resolves to initial_tag
        labeled_set:
            The filenames of the samples in the labeled set, List[str]
        unlabeled_set:
            The filenames of the samples in the unlabeled set, List[str]

    Examples:
        >>> # set the token and dataset id
        >>> token = '123'
        >>> dataset_id = 'XYZ'
        >>>
        >>> # create an active learning agent
        >>> client = ApiWorkflowClient(token, dataset_id)
        >>> agent = ActiveLearningAgent(client)
        >>>
        >>> # make an initial active learning query
        >>> sampler_config = SamplerConfig(n_samples=100, name='initial-set')
        >>> initial_set = agent.query(sampler_config)
        >>> unlabeled_set = agent.unlabeled_set
        >>>
        >>> # train and evaluate a model on the initial set
        >>> # make predictions on the unlabeled set (keep ordering of filenames)
        >>>
        >>> # create active learning scorer
        >>> scorer = ScorerClassification(predictions)
        >>>
        >>> # make a second active learning query
        >>> sampler_config = SamplerConfig(n_samples=200, name='second-set')
        >>> second_set = agent.query(sampler_config, scorer)

    """

    def __init__(self, api_workflow_client: ApiWorkflowClient, query_tag_name: str = None,
                 preselected_tag_name: str = None):

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

        if not hasattr(self, "bitmask_labeled_set"):
            self.bitmask_labeled_set = BitMask.from_hex("0x0")  # empty labeled set
            self.bitmask_added_set = BitMask.from_hex("0x0")  # empty added set
        if self.preselected_tag_id is not None:  # else the default values (empty labeled and added set) are kept
            if preselected_tag_data is None:  # if it is not passed as argument, it must be loaded from the API
                preselected_tag_data = self.api_workflow_client.tags_api.get_tag_by_tag_id(
                    self.api_workflow_client.dataset_id, tag_id=self.preselected_tag_id)
            new_bitmask_labeled_set = BitMask.from_hex(preselected_tag_data.bit_mask_data)
            self.bitmask_added_set = new_bitmask_labeled_set - self.bitmask_labeled_set
            self.bitmask_labeled_set = new_bitmask_labeled_set

        if self.query_tag_id is None:
            bitmask_query_tag = BitMask.from_length(len(self.api_workflow_client.filenames_on_server))
        else:
            query_tag_data = self.api_workflow_client.tags_api.get_tag_by_tag_id(
                self.api_workflow_client.dataset_id, tag_id=self.query_tag_id)
            bitmask_query_tag = BitMask.from_hex(query_tag_data.bit_mask_data)
        self.bitmask_unlabeled_set = bitmask_query_tag - self.bitmask_labeled_set

        self.labeled_set = self.bitmask_labeled_set.masked_select_from_list(self.api_workflow_client.filenames_on_server)
        self.added_set = self.bitmask_added_set.masked_select_from_list(self.api_workflow_client.filenames_on_server)
        self.unlabeled_set = self.bitmask_unlabeled_set.masked_select_from_list(self.api_workflow_client.filenames_on_server)

    def query(self, sampler_config: SamplerConfig, al_scorer: Scorer = None) -> Tuple[List[str], List[str]]:
        """Performs an active learning query.

        As part of it, the self.labeled_set and self.unlabeled_set are updated
        and can be used for the next step.

        Args:
            sampler_config:
                The sampling configuration.
            al_scorer:
                An instance of a class inheriting from Scorer, e.g. a ClassificationScorer.

        Returns:
            The filenames of the samples in the new labeled_set
            and the filenames of the samples chosen by the sampler.
            This added_set was added to the old labeled_set
            to form the new labeled_set.

        """
        # check input
        if sampler_config.n_samples < len(self.labeled_set):
            warnings.warn("ActiveLearningAgent.query: The number of samples which should be sampled "
                          "including the current labeled set "
                          "(sampler_config.n_samples) "
                          "is smaller than the number of samples in the current labeled set."
                          "Skipping the sampling and returning the old labeled_set and"
                          "no ne filenames.")
            return self.labeled_set, []

        # calculate scores
        if al_scorer is not None:
            no_unlabeled_samples = len(self.unlabeled_set)
            no_samples_with_predictions = len(al_scorer.model_output)
            if no_unlabeled_samples != no_samples_with_predictions:
                raise ValueError(f"The scorer must have exactly as many samples as in the unlabeled set,"
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

        return self.labeled_set, self.added_set
