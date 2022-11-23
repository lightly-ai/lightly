from abc import ABC
from typing import Optional, List

from lightly.openapi_generated.swagger_client import TaskType


class PredictionSingletonRepr(ABC):
    def __init__(
        self,
        type: str,
        taskName: str,
        categoryId: int,
        score: float,
        cropDatasetId: Optional[str] = None,
        cropSampleId: Optional[str] = None,
    ):
        self.type = type
        self.taskName = taskName
        self.categoryId = categoryId
        self.score = score
        if cropDatasetId is not None:
            self.cropDatasetId = cropDatasetId
        if cropSampleId is not None:
            self.cropSampleId = cropSampleId

    def to_dict(self):
        return vars(self)


class PredictionSingletonClassificationRepr(PredictionSingletonRepr):
    def __init__(
        self,
        taskName: str,
        categoryId: int,
        score: float,
        probabilities: Optional[List[float]] = None,
    ):
        PredictionSingletonRepr.__init__(
            self,
            type=TaskType.CLASSIFICATION,
            taskName=taskName,
            categoryId=categoryId,
            score=score,
        )
        self.probabilities = probabilities
