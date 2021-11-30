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
        query_set:
            Set of filenames corresponding to samples which can possibly be selected.
            Set to all samples in the query tag or to the whole dataset by default.
        labeled_set:
            Set of filenames corresponding to samples in the labeled set.
            Set to all samples in the preselected tag or to an empty list by default.
        unlabeled_set:
            Set of filenames corresponding to samples which are in the query set
            but not in the labeled set.
        added_set:
            Set of filenames corresponding to samples which were added to the 
            labeled set in the last query.
            
            Raises:
                RuntimeError: If executed before a query.

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
        >>> agent.query(sampler_config)
        >>> initial_set = agent.labeled_set
        >>>
        >>> # train and evaluate a model on the initial set
        >>> # make predictions on the query set:
        >>> query_set = agent.query_set
        >>> # important:
        >>> # be sure to keep the order of the query set when you make predictions
        >>>
        >>> # create active learning scorer
        >>> scorer = ScorerClassification(predictions)
        >>>
        >>> # make a second active learning query
        >>> sampler_config = SamplerConfig(n_samples=200, name='second-set')
        >>> agent.query(sampler_config, scorer)
        >>> added_set = agent.added_set # access only the samples added by this query

    """

    def __init__(self,
                 api_workflow_client: ApiWorkflowClient,
                 query_tag_name: str = 'initial-tag',
                 preselected_tag_name: str = None):

        self.api_workflow_client = api_workflow_client

        # set the query_tag_id and preselected_tag_id
        self._query_tag_id = None
        self._preselected_tag_id = None

        # build lookup table for tag_name to tag_id
        tag_name_id_dict = {}
        for tag in self.api_workflow_client._get_all_tags():
            tag_name_id_dict[tag.name] = tag.id
        # use lookup table to set ids
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
            RuntimeError: If executed before a query.

        """
        # the added set only exists after a query
        if self._old_preselected_tag_bitmask is None:
            raise RuntimeError('Cannot compute \"added set\" before querying.')
        # added set is new preselected set minus the old one
        added_tag_bitmask = self._preselected_tag_bitmask - self._old_preselected_tag_bitmask
        return added_tag_bitmask.masked_select_from_list(
            self.api_workflow_client.filenames_on_server
        )


    def upload_scores(self, al_scorer: Scorer):
        """Computes and uploads active learning scores to the Lightly webapp.

        Args:
            al_scorer:
                An instance of a class inheriting from Scorer, e.g. a ClassificationScorer.

        """
        # calculate active learning scores
        al_scores_dict = al_scorer.calculate_scores()

        # Check if the length of the query_set and each of the scores are the same
        no_query_samples = len(self.query_set)
        for score in al_scores_dict.values():
            no_query_samples_with_scores = len(score)
            if no_query_samples != no_query_samples_with_scores:
                raise ValueError(
                    f'Number of query samples ({no_query_samples}) must match '
                    f'the number of predictions ({no_query_samples_with_scores})!'
                )
        self.api_workflow_client.upload_scores(al_scores_dict, self._query_tag_id)


    def query(self,
              sampler_config: SamplerConfig,
              al_scorer: Scorer = None):
        """Performs an active learning query.

        First the active learning scores are computed and uploaded,
        then the sampling query is performed.
        After the query, the labeled set is updated to contain all selected samples,
        the added set is recalculated as (new labeled set - old labeled set), and
        the query set stays the same.

        Args:
            sampler_config:
                The sampling configuration.
            al_scorer:
                An instance of a class inheriting from Scorer, e.g. a ClassificationScorer.

        """

        # handle illogical stopping condition
        if sampler_config.n_samples < len(self.labeled_set):
            warnings.warn(
                f'ActiveLearningAgent.query: The number of samples ({sampler_config.n_samples}) is '
                f'smaller than the number of preselected samples ({len(self.labeled_set)}).'
                'Skipping the active learning query.'
            )
            return

        if al_scorer:
            self.upload_scores(al_scorer)

        # perform the sampling
        new_tag_data = self.api_workflow_client.sampling(
            sampler_config=sampler_config,
            preselected_tag_id=self._preselected_tag_id,
            query_tag_id=self._query_tag_id
        )

        # update the old preselected_tag
        self._old_preselected_tag_bitmask = self._preselected_tag_bitmask
        # set the newly chosen tag as the new preselected_tag
        self._preselected_tag_id = new_tag_data.id
        self._preselected_tag_bitmask = self._get_preselected_tag_bitmask()
