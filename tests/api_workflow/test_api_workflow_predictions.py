from unittest.mock import MagicMock, call

from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client.api import PredictionsApi
from lightly.openapi_generated.swagger_client.models import (
    PredictionSingletonClassification,
    PredictionTaskSchema,
    PredictionTaskSchemaCategory,
    TaskType,
)


def test_create_or_update_prediction_task_schema() -> None:
    mocked_client = MagicMock(spec=ApiWorkflowClient)
    mocked_client.dataset_id = "some_dataset_id"
    mocked_client._predictions_api = MagicMock(spec_set=PredictionsApi)

    schema = PredictionTaskSchema.from_dict(
        {
            "name": "my-object-detection",
            "type": TaskType.OBJECT_DETECTION,
            "categories": [
                PredictionTaskSchemaCategory(id=0, name="dog").to_dict(),
                PredictionTaskSchemaCategory(id=1, name="cat").to_dict(),
            ],
        }
    )
    timestamp = 1234
    ApiWorkflowClient.create_or_update_prediction_task_schema(
        self=mocked_client,
        schema=schema,
        prediction_version_id=timestamp,
    )

    mocked_client._predictions_api.create_or_update_prediction_task_schema_by_dataset_id.assert_called_once_with(
        prediction_task_schema=schema,
        dataset_id=mocked_client.dataset_id,
        prediction_uuid_timestamp=timestamp,
    )


def test_create_or_update_prediction() -> None:
    mocked_client = MagicMock(spec=ApiWorkflowClient)
    mocked_client.dataset_id = "some_dataset_id"
    mocked_client._predictions_api = MagicMock(spec_set=PredictionsApi)

    prediction_singletons = [
        PredictionSingletonClassification(
            type="CLASSIFICATION",
            taskName="my-task",
            categoryId=1,
            score=0.9,
            probabilities=[0.1, 0.2, 0.3, 0.4],
        )
    ]

    sample_id = "some_sample_id"
    timestamp = 1234
    ApiWorkflowClient.create_or_update_prediction(
        self=mocked_client,
        sample_id=sample_id,
        prediction_singletons=prediction_singletons,
        prediction_version_id=timestamp,
    )

    mocked_client._predictions_api.create_or_update_prediction_by_sample_id.assert_called_once_with(
        prediction_singleton=prediction_singletons,
        dataset_id=mocked_client.dataset_id,
        sample_id=sample_id,
        prediction_uuid_timestamp=timestamp,
    )
