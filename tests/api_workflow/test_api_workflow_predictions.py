import time
from unittest.mock import MagicMock, _Call, call

from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient
from lightly.api.prediction_singletons import PredictionSingletonClassificationRepr
from lightly.openapi_generated.swagger_client import (
    PredictionTaskSchema,
    TaskType,
    PredictionTaskSchemaCategory,
    PredictionsApi,
)


def test_create_or_update_prediction_task_schema() -> None:
    mocked_client = MagicMock(spec=ApiWorkflowClient)
    mocked_client.dataset_id = "some_dataset_id"
    mocked_client._predictions_api = MagicMock(spec_set=PredictionsApi)

    schema = PredictionTaskSchema(
        name="my-object-detection",
        type=TaskType.OBJECT_DETECTION,
        categories=[
            PredictionTaskSchemaCategory(id=0, name="dog"),
            PredictionTaskSchemaCategory(id=1, name="cat"),
        ],
    )
    timestamp = int(time.time())
    ApiWorkflowClient.create_or_update_prediction_task_schema(
        self=mocked_client,
        schema=schema,
        prediction_uuid_timestamp=timestamp,
    )

    mocked_client._predictions_api.create_or_update_prediction_task_schema_by_dataset_id.assert_called_once_with(
        body=schema,
        dataset_id=mocked_client.dataset_id,
        prediction_uuid_timestamp=timestamp,
    )


def test_create_or_update_prediction() -> None:
    mocked_client = MagicMock(spec=ApiWorkflowClient)
    mocked_client.dataset_id = "some_dataset_id"
    mocked_client._predictions_api = MagicMock(spec_set=PredictionsApi)

    prediction_singletons = [
        PredictionSingletonClassificationRepr(
            taskName="my-task",
            categoryId=1,
            score=0.9,
            probabilities=[0.1, 0.2, 0.3, 0.4],
        )
    ]
    expected_upload_prediction_singletons = [
        singleton.to_dict() for singleton in prediction_singletons
    ]

    sample_id = "some_sample_id"
    timestamp = int(time.time())
    ApiWorkflowClient.create_or_update_prediction(
        self=mocked_client,
        sample_id=sample_id,
        prediction_singletons=prediction_singletons,
        prediction_version_timestamp=timestamp,
    )

    mocked_client._predictions_api.create_or_update_prediction_by_sample_id.assert_called_once_with(
        body=expected_upload_prediction_singletons,
        dataset_id=mocked_client.dataset_id,
        sample_id=sample_id,
        prediction_uuid_timestamp=timestamp,
    )


def test_create_or_update_prediction() -> None:
    mocked_client = MagicMock(spec=ApiWorkflowClient)
    mocked_client.dataset_id = "some_dataset_id"
    mocked_client._predictions_api = MagicMock(spec_set=PredictionsApi)

    prediction_singletons = [
        PredictionSingletonClassificationRepr(
            taskName="my-task",
            categoryId=1,
            score=0.9,
            probabilities=[0.1, 0.2, 0.3, 0.4],
        )
    ]
    expected_upload_prediction_singletons = [
        singleton.to_dict() for singleton in prediction_singletons
    ]

    sample_id = "some_sample_id"
    timestamp = int(time.time())
    ApiWorkflowClient.create_or_update_prediction(
        self=mocked_client,
        sample_id=sample_id,
        prediction_singletons=prediction_singletons,
        prediction_version_timestamp=timestamp,
    )

    mocked_client._predictions_api.create_or_update_prediction_by_sample_id.assert_called_once_with(
        body=expected_upload_prediction_singletons,
        dataset_id=mocked_client.dataset_id,
        sample_id=sample_id,
        prediction_uuid_timestamp=timestamp,
    )


def test_create_or_update_predictions() -> None:
    mocked_client = MagicMock(spec=ApiWorkflowClient).return_value
    mocked_client.dataset_id = "some_dataset_id"

    sample_id_to_prediction_singletons_dummy = {
        f"sample_id_{i}": [
            PredictionSingletonClassificationRepr(
                taskName="my-task",
                categoryId=i % 4,
                score=0.9,
                probabilities=[0.1, 0.2, 0.3, 0.4],
            )
        ]
        for i in range(4)
    }

    timestamp = int(time.time())
    ApiWorkflowClient.create_or_update_predictions(
        self=mocked_client,
        sample_id_to_prediction_singletons=sample_id_to_prediction_singletons_dummy,
        prediction_version_timestamp=timestamp,
    )

    expected_calls = [
        call(
            sample_id=sample_id,
            prediction_singletons=singletons,
            prediction_version_timestamp=timestamp,
        )
        for sample_id, singletons in sample_id_to_prediction_singletons_dummy.items()
    ]
    mocked_client.create_or_update_prediction.assert_has_calls(
        calls=expected_calls, any_order=True
    )