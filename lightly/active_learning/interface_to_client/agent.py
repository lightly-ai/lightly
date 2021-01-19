from typing import *

import numpy as np

from lightly.active_learning.interface_to_client.sampler_config import SamplerConfig
from lightly.active_learning.scorers.scorer import Scorer

from lightly.active_learning.interface_to_api.active_learning_step import sampling_step_to_api


class ActiveLearningAgent:
    def __init__(self, token: str = '', dataset_id: str = '', path_to_embeddings: str = ''):
        self.path_to_embeddings = path_to_embeddings
        self.dataset_id = dataset_id
        self.token = token

    def sample(self, sampler_config: SamplerConfig, al_scorer: Scorer = None, labelled_ids: List[int] = []):
        if al_scorer is not None:
            scores = al_scorer._calculate_scores()
        else:
            scores = dict()
        chosen_samples = sampling_step_to_api(sampler_config, scores, labelled_ids)
        return chosen_samples
