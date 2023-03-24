from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from lightly.active_learning import raise_active_learning_deprecation_warning


class Scorer(ABC):

    def __init__(self):
        raise_active_learning_deprecation_warning()

    @abstractmethod
    def calculate_scores(self) -> Dict[str, np.ndarray]:
        """Calculates and returns active learning scores in a dictionary."""
        ...

    @classmethod
    @abstractmethod
    def score_names(cls) -> List[str]:
        """Returns the names of the calculated active learning scores"""
        ...
