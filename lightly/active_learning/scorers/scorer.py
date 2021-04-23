from typing import *
import abc

import numpy as np


class Scorer():
    def __init__(self, model_output):
        self.model_output = model_output

    def calculate_scores(self) -> Dict[str, np.ndarray]:
        """Calculates and returns active learning scores in a dictionary.
        """
        raise NotImplementedError

    @classmethod
    def score_names(cls) -> List[str]:
        """Returns the names of the calculated active learning scores
        """
