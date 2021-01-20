from typing import *

import numpy as np

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.scorer import Scorer

from lightly.api.upload import upload_embeddings_from_csv
from lightly.api.active_learning import upload_scores_to_api, sampling_request_to_api


class ActiveLearningAgent:
    def __init__(self, token: str = '', dataset_id: str = '', path_to_embeddings: str = ''):

        upload_embeddings_from_csv(path_to_embeddings, dataset_id, token)

        self.dataset_id = dataset_id
        self.token = token

    def sample(self, sampler_config: SamplerConfig, al_scorer: Scorer = None, labelled_ids: List[int] = []):
        if al_scorer is not None:
            scores = al_scorer._calculate_scores()

        use_mock = True
        if use_mock:
            return np.random.randint(0, len(scores), sampler_config.batch_size)

        if al_scorer is not None:
            upload_scores_to_api(scores)

        chosen_samples = sampling_request_to_api(sampler_config=sampler_config, labelled_ids=labelled_ids)

        return chosen_samples
