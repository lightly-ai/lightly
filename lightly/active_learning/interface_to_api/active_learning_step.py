from typing import *

import numpy as np

from lightly.active_learning.interface_to_client.sampler_config import SamplerConfig


def sampling_step_to_api(sampler_config: SamplerConfig, scores: Dict[str, np.ndarray] = None, labelled_ids: List[int] = []):
    use_mock = True
    if use_mock:
        return np.random.randint(0, len(scores), sampler_config.batch_size)

    if scores is not None:
        upload_scores_to_api(scores)

    sampling_request_to_api(sampler_config, labelled_ids)


def upload_scores_to_api(scores: Dict[str, Iterable[float]]):
    raise NotImplementedError


def sampling_request_to_api(sampler_config: SamplerConfig, labelled_ids: List[int] = 0):
    raise NotImplementedError
