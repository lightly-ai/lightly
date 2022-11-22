from abc import ABC
from typing import Optional, List

from lightly.openapi_generated.swagger_client import TaskType


class PredictionSingletonRepr(ABC):
    type: str
    taskName: str
    cropDatasetId: Optional[str] = None
    cropSampleId: Optional[str] = None
    categoryId: int
    score: float


class PredictionSingletonClassificationRepr(PredictionSingletonRepr):
    probabilities: Optional[List[float]]

    def __init__(self, taskName: str, categoryId: int, score: float, probabilities: Optional[List[float]] = None):
        self.type = TaskType.CLASSIFICATION
        self.taskName = taskName
        self.categoryId = categoryId
        self.score = score
        self.probabilities = probabilities
