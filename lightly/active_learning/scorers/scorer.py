from typing import Dict, List

import numpy as np


class Scorer:
    def calculate_scores(self) -> Dict[str, np.ndarray]:
        """Calculates and returns active learning scores in a dictionary."""
        raise NotImplementedError

    @classmethod
    def score_names(cls) -> List[str]:
        """Returns the names of the calculated active learning scores"""
        raise NotImplementedError
