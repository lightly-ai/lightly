from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class Scorer(ABC):
    @abstractmethod
    def calculate_scores(self) -> Dict[str, np.ndarray]:
        """Calculates and returns active learning scores in a dictionary."""
        ...

    @classmethod
    @abstractmethod
    def score_names(cls) -> List[str]:
        """Returns the names of the calculated active learning scores"""
        ...
