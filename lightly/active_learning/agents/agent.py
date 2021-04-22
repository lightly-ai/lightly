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
        TODO

    Examples:
        TODO: rework
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

    def __init__(self,
                 api_workflow_client: ApiWorkflowClient,
                 query_tag_name: str = None,
                 preselected_tag_name: str = None):

        self.api_workflow_client = api_workflow_client

        # set the query_tag_id and preselected_tag_id
        self._query_tag_id = None
        self._preselected_tag_id = None
        if query_tag_name is not None or preselected_tag_name is not None:
            # build lookup table for tag_name to tag_id
            tag_name_id_dict = {}
            for tag in self.api_workflow_client._get_all_tags():
                tag_name_id_dict[tag.name] = tag.id
            # use lookup table to set ids
            if query_tag_name is not None:
                self._query_tag_id = tag_name_id_dict[query_tag_name]
            if preselected_tag_name is not None:
                self._preselected_tag_id = tag_name_id_dict[preselected_tag_name]

        # set the filename lists based on preselected and query tag
        self._query_tag_bitmask = self._get_query_tag_bitmask()
        self._preselected_tag_bitmask = self._get_preselected_tag_bitmask()
        # keep track of the last preselected tag to compute added samples
        self._old_preselected_tag_bitmask = None


    def _get_query_tag_bitmask(self):
        """Initializes the query tag bitmask.

        """
        if self._query_tag_id is None:
            # if not specified, all samples belong to the query tag
            query_tag_bitmask = BitMask.from_length(
                len(self.api_workflow_client.filenames_on_server)
            )
        else:
            # get query tag from api and set bitmask accordingly
            query_tag_data = self.api_workflow_client.tags_api.get_tag_by_tag_id(
                self.api_workflow_client.dataset_id,
                tag_id=self._query_tag_id
            )
            query_tag_bitmask = BitMask.from_hex(query_tag_data.bit_mask_data)

        return query_tag_bitmask

    def _get_preselected_tag_bitmask(self):
        """Initializes the preselected tag bitmask.

        """
        if self._preselected_tag_id is None:
            # if not specified, no samples belong to the preselected tag
            preselected_tag_bitmask = BitMask.from_hex('0x0')
        else:
            # get preselected tag from api and set bitmask accordingly
            preselected_tag_data = self.api_workflow_client.tags_api.get_tag_by_tag_id(
                self.api_workflow_client.dataset_id,
                tag_id=self._preselected_tag_id
            )
            preselected_tag_bitmask = BitMask.from_hex(preselected_tag_data.bit_mask_data)

        return preselected_tag_bitmask

    @property
    def query_set(self):
        """List of filenames for which to calculate active learning scores.

        """
        return self._query_tag_bitmask.masked_select_from_list(
            self.api_workflow_client.filenames_on_server
        )

    @property
    def labeled_set(self):
        """List of filenames indicating selected samples.

        """
        return self._preselected_tag_bitmask.masked_select_from_list(
            self.api_workflow_client.filenames_on_server
        )

    @property
    def unlabeled_set(self):
        """List of filenames which belong to the query set but are not selected.

        """
        # unlabeled set is the query set minus the preselected set
        unlabeled_tag_bitmask = self._query_tag_bitmask - self._preselected_tag_bitmask
        return unlabeled_tag_bitmask.masked_select_from_list(
            self.api_workflow_client.filenames_on_server
        )

    @property
    def added_set(self):
        """List of filenames of newly added samples (in the last query).

        Raises:
            RuntimeError if executed before a query.

        """
        # the added set only exists after a query
        if self._old_preselected_tag_bitmask is None:
            raise RuntimeError('Cannot compute \"added set\" before querying.')
        # added set is new preselected set minus the old one
        added_tag_bitmask = self._preselected_tag_bitmask - self._old_preselected_tag_bitmask
        return added_tag_bitmask.masked_select_from_list(
            self.api_workflow_client.filenames_on_server
        )


    def query(self,
              sampler_config: SamplerConfig,
              al_scorer: Scorer = None) -> Tuple[List[str], List[str]]:
        """Performs an active learning query.

        TODO: what happens here?

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

        # handle illogical stopping condition
        if sampler_config.n_samples < len(self.labeled_set):
            warnings.warn(
                f'ActiveLearningAgent.query: The number of samples ({sampler_config.n_samples}) is '
                f'smaller than the number of preselected samples ({len(self.labeled_set)}).'
                'Skipping the active learning query and returning the previous labeled set.'
            )
            return self.labeled_set, []

        # calculate active learning scores
        scores_dict = None
        if al_scorer is not None:
            no_query_samples = len(self.query_set)
            no_query_samples_with_scores = len(al_scorer.model_output)
            if no_query_samples != no_query_samples_with_scores:
                raise ValueError(
                    f'Number of query samples ({no_query_samples}) must match '
                    f'the number of predictions ({no_query_samples_with_scores})!'
                )
            scores_dict = al_scorer.calculate_scores()

        # perform the sampling
        new_tag_data = self.api_workflow_client.sampling(
            sampler_config=sampler_config,
            al_scores=scores_dict,
            preselected_tag_id=self._preselected_tag_id,
            query_tag_id=self._query_tag_id
        )

        # update the old preselected_tag
        self._old_preselected_tag_bitmask = self._preselected_tag_bitmask
        # set the newly chosen tag as the new preselected_tag
        self._preselected_tag_id = new_tag_data.id
        self._preselected_tag_bitmask = self._get_preselected_tag_bitmask()

        return self.labeled_set, self.added_set
