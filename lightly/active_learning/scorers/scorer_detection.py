from typing import *

import numpy as np

from active_learning.scorers.scorer import ALScorer


class ALScoreComputerDetection(ALScorer):
    def __init__(self, model_output: List[List[np.ndarra]]):
        # TODO check model_output

        self.model_output = model_output

    def _calculate_scores(self) -> Dict[str, np.ndarray]:
        scores = dict()

        raise NotImplementedError
        return scores

