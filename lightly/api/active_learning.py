""" Upload to Lightly API """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import *

from lightly.active_learning.config.sampler_config import SamplerConfig


def upload_scores_to_api(scores: Dict[str, Iterable[float]]):
    raise NotImplementedError


def sampling_request_to_api(sampler_config: SamplerConfig, labelled_ids: List[int] = 0):
    raise NotImplementedError
