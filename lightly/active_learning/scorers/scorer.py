from typing import *
import abc

import numpy as np


class ALScorer():
    def __init__(self, model_output):
        raise NotImplementedError

    def _calculate_scores(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError
