from typing import *
import abc

import numpy as np


class Scorer():
    def __init__(self, model_output):
        self.model_output = model_output

    def calculate_scores(self) -> Dict[str, np.ndarray]:
        """Calculates and returns the active learning scores

        Which scores are calculated depends on the implementation
        of this parent class by the child classes.
        Returns:
            A dictionary mapping from the score name (as string)
            to the scores (as a single-dimensional numpy array).
        """
        raise NotImplementedError
