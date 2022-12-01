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
        self.cropDatasetId = cropDatasetId
        self.cropSampleId = cropSampleId

    def to_dict(self):
        return {key: value for key, value in vars(self).items() if value is not None}


class PredictionSingletonClassificationRepr(PredictionSingletonRepr):
    def __init__(
        self,
        taskName: str,
        categoryId: int,
        score: float,
        probabilities: Optional[List[float]] = None,
    ):
        super().__init__(
            type=TaskType.CLASSIFICATION,
            taskName=taskName,
            categoryId=categoryId,
            score=score,
        )
        self.probabilities = probabilities


class PredictionSingletonObjectDetectionRepr(PredictionSingletonRepr):
    def __init__(
        self,
        taskName: str,
        categoryId: int,
        score: float,
        cropDatasetId: str,
        cropSampleId: str,
        bbox: List[int],
        probabilities: Optional[List[float]] = None,
    ):
        super().__init__(
            type=TaskType.OBJECT_DETECTION,
            taskName=taskName,
            categoryId=categoryId,
            score=score,
            cropDatasetId=cropDatasetId,
            cropSampleId=cropSampleId,
        )
        self.bbox = bbox
        self.probabilities = probabilities


class PredictionSingletonSemanticSegmentationRepr(PredictionSingletonRepr):
    def __init__(
        self,
        taskName: str,
        categoryId: int,
        score: float,
        cropDatasetId: str,
        cropSampleId: str,
        segmentation: str,
        probabilities: Optional[List[float]] = None,
    ):
        super().__init__(
            type=TaskType.SEMANTIC_SEGMENTATION,
            taskName=taskName,
            categoryId=categoryId,
            score=score,
            cropDatasetId=cropDatasetId,
            cropSampleId=cropSampleId,
        )
        self.segmentation = segmentation
        self.probabilities = probabilities


# Not used
class PredictionSingletonInstanceSegmentationRepr(PredictionSingletonRepr):
    def __init__(
        self,
        taskName: str,
        categoryId: int,
        score: float,
        cropDatasetId: str,
        cropSampleId: str,
        segmentation: str,
        probabilities: Optional[List[float]] = None,
    ):
        super().__init__(
            type=TaskType.INSTANCE_SEGMENTATION,
            taskName=taskName,
            categoryId=categoryId,
            score=score,
            cropDatasetId=cropDatasetId,
            cropSampleId=cropSampleId,
        )
        self.segmentation = segmentation
        self.probabilities = probabilities


# Not used
class PredictionSingletonKeypointDetectionRepr(PredictionSingletonRepr):
    def __init__(
        self,
        taskName: str,
        categoryId: int,
        score: float,
        keypoints: List[int],
        probabilities: Optional[List[float]] = None,
    ):
        super().__init__(
            type=TaskType.KEYPOINT_DETECTION,
            taskName=taskName,
            categoryId=categoryId,
            score=score,
        )
        self.keypoints = keypoints
        self.probabilities = probabilities
