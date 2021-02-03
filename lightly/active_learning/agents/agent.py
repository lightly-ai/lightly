import os
from typing import *
import time

import numpy as np

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.scorer import Scorer
from lightly.api.api_workflow import ApiWorkflow

from lightly.api.upload import upload_csv
from lightly.api.utils import create_api_client
from lightly.api.tags import get_tag_by_tag_id
from lightly.openapi_generated.swagger_client import JobStatusData, JobState
from lightly.api.bitmask import BitMask


class ActiveLearningAgent:
    def __init__(self, token: str = '', dataset_id: str = '', initial_tag: str = 'initial_tag',
                 path_to_embeddings: str = '', host: str = 'https://api-dev.lightly.ai'):

        os.environ['LIGHTLY_SERVER_LOCATION'] = host
        embedding_id = upload_csv(path_to_embeddings, dataset_id, token)

        self.api_workflow = ApiWorkflow(host=host, token=token, dataset_id=dataset_id, embedding_id=embedding_id)

    def sample(self, sampler_config: SamplerConfig, al_scorer: Scorer = None, preselected_tag_id: str = None) \
            -> List[int]:

        # calculate scores
        if al_scorer is not None:
            scores = al_scorer._calculate_scores()
        else:
            scores = None

        # perform the sampling
        new_tag_data = self.api_workflow.sampling(sampler_config=sampler_config, al_scores=scores)

        # extract the chosen samples as List[int]
        chosen_samples = BitMask.from_bin(new_tag_data.bit_mask_data).to_indices()

        return chosen_samples
