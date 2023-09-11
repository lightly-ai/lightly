from typing import Sequence

from lightly.openapi_generated.swagger_client.models import (
    PredictionSingleton,
    PredictionTaskSchema,
)


class _PredictionsMixin:
    def create_or_update_prediction_task_schema(
        self,
        schema: PredictionTaskSchema,
        prediction_version_id: int = -1,
    ) -> None:
        """Creates or updates the prediction task schema.

        Args:
            schema:
                The prediction task schema.
            prediction_version_id:
                A numerical ID (e.g., timestamp) to distinguish different predictions of different model versions.
                Use the same ID if you don't require versioning or if you wish to overwrite the previous schema.

        Example:
          >>> import time
          >>> from lightly.api import ApiWorkflowClient
          >>> from lightly.openapi_generated.swagger_client.models import (
          >>>     PredictionTaskSchema,
          >>>     TaskType,
          >>>     PredictionTaskSchemaCategory,
          >>> )
          >>>
          >>> client = ApiWorkflowClient(
          >>>     token="MY_LIGHTLY_TOKEN", dataset_id="MY_DATASET_ID"
          >>> )
          >>>
          >>> schema = PredictionTaskSchema(
          >>>     name="my-object-detection",
          >>>     type=TaskType.OBJECT_DETECTION,
          >>>     categories=[
          >>>         PredictionTaskSchemaCategory(id=0, name="dog"),
          >>>         PredictionTaskSchemaCategory(id=1, name="cat"),
          >>>     ],
          >>> )
          >>> client.create_or_update_prediction_task_schema(schema=schema)

        :meta private:  # Skip docstring generation
        """
        self._predictions_api.create_or_update_prediction_task_schema_by_dataset_id(
            prediction_task_schema=schema,
            dataset_id=self.dataset_id,
            prediction_uuid_timestamp=prediction_version_id,
        )

    def create_or_update_prediction(
        self,
        sample_id: str,
        prediction_singletons: Sequence[PredictionSingleton],
        prediction_version_id: int = -1,
    ) -> None:
        """Creates or updates predictions for one specific sample.

        Args:
            sample_id
                The ID of the sample.

            prediction_version_id:
                A numerical ID (e.g., timestamp) to distinguish different predictions of different model versions.
                Use the same id if you don't require versioning or if you wish to overwrite the previous schema.
                This ID must match the ID of a prediction task schema.

            prediction_singletons:
                Predictions to be uploaded for the designated sample.

        :meta private:  # Skip docstring generation
        """
        self._predictions_api.create_or_update_prediction_by_sample_id(
            prediction_singleton=prediction_singletons,
            dataset_id=self.dataset_id,
            sample_id=sample_id,
            prediction_uuid_timestamp=prediction_version_id,
        )
