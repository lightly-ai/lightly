from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union, Optional, Tuple

import tqdm

from lightly.api.prediction_singletons import (
    PredictionSingletonClassificationRepr,
    PredictionSingletonRepr,
)
from lightly.api.utils import retry
from lightly.openapi_generated.swagger_client import (
    PredictionTaskSchema,
    SamplePartialMode,
    PredictionSingletonClassification,
    PredictionSingletonObjectDetection,
    PredictionSingletonInstanceSegmentation,
    PredictionSingletonKeypointDetection,
)


class _PredictionsMixin:
    def create_or_update_prediction_task_schema(
        self,
        schema: PredictionTaskSchema,
        prediction_uuid_timestamp: int,
    ) -> None:
        """Creates or updates the prediction task schema

        Args:
            schema:
                The prediction task schema.
            prediction_uuid_timestamp:
                This timestamp is used as a key to distinguish different predictions for the same sample.
                Get it e.g. via 'int(time.time())'

        Example:
          >>> import time
          >>> from lightly.api import ApiWorkflowClient
          >>> from lightly.openapi_generated.swagger_client import (
          >>>     PredictionTaskSchema,
          >>>     TaskType,
          >>>     PredictionTaskSchemaCategory,
          >>> )
          >>>
          >>> client = ApiWorkflowClient(
          >>>     token="MY_LIGHTLY_TOKEN", dataset_id="MY_DATASET_ID"
          >>> )
          >>>
          >>> timestamp = int(time.time())
          >>> schema = PredictionTaskSchema(
          >>>     name="my-object-detection",
          >>>     type=TaskType.OBJECT_DETECTION,
          >>>     categories=[
          >>>         PredictionTaskSchemaCategory(id=0, name="dog"),
          >>>         PredictionTaskSchemaCategory(id=1, name="cat"),
          >>>     ],
          >>> )
          >>> client.create_or_update_prediction_task_schema(
          >>>     schema=schema, prediction_uuid_timestamp=timestamp
          >>> )


        """
        self._predictions_api.create_or_update_prediction_task_schema_by_dataset_id(
            body=schema,
            dataset_id=self.dataset_id,
            prediction_uuid_timestamp=prediction_uuid_timestamp,
        )

    def create_or_update_predictions(
        self,
        sample_id_to_prediction_singletons: Dict[str, List[PredictionSingletonRepr]],
        prediction_version_timestamp: int,
        progress_bar: Optional[tqdm.tqdm] = None,
        max_workers: int = 8,
    ) -> None:
        """Creates or updates the predictions for specific samples

        Args:
            sample_id_to_prediction_singletons
                A mapping from the sample_id of the sample to its corresponding prediction singletons.
                The singletons can be from different tasks and different types.

            prediction_version_timestamp:
                This timestamp is used as a key to distinguish different predictions for the same sample.
                Get it e.g. via 'int(time.time())'.

            progress_bar:
                Tqdm progress bar to show how many prediction files have already been
                uploaded.

            max_workers:
                Maximum number of workers uploading predictions in parallel.

        Example:
          >>> import time
          >>> from tqdm import tqdm
          >>> from lightly.api import ApiWorkflowClient
          >>> from lightly.openapi_generated.swagger_client import (
          >>>     PredictionTaskSchema,
          >>>     TaskType,
          >>>     PredictionTaskSchemaCategory,
          >>> )
          >>> from lightly.api.prediction_singletons import PredictionSingletonClassificationRepr
          >>>
          >>> client = ApiWorkflowClient(
          >>>     token="MY_LIGHTLY_TOKEN", dataset_id="MY_DATASET_ID"
          >>> )
          >>>
          >>> timestamp = int(time.time())
          >>> filenames = client.get_filenames()
          >>> filename_to_prediction_singletons_dummy = {
          >>>     filename: [PredictionSingletonClassificationRepr(taskName="my-task", categoryId=i%4, score=0.9, probabilities=[0.1, 0.2, 0.3, 0.4])]
          >>>     for i, filename in enumerate(filenames)
          >>> }
          >>> client.create_or_update_predictions(
          >>>     filename_to_prediction_singletons_dummy,
          >>>     prediction_uuid_timestamp=timestamp,
          >>>     progress_bar=tqdm(desc="Uploading predictions", total=len(filenames), unit=" predictions")
          >>> )


        """

        # handle the case where len(filename_to_sample_id) < max_workers
        max_workers = min(len(sample_id_to_prediction_singletons), max_workers)
        max_workers = max(max_workers, 1)

        def upload_prediction(
            sample_id_prediction_singletons_tuple: Tuple[
                str, List[PredictionSingletonRepr]
            ]
        ) -> None:
            (sample_id, prediction_singletons) = sample_id_prediction_singletons_tuple
            self.create_or_update_prediction(
                sample_id=sample_id,
                prediction_singletons=prediction_singletons,
                prediction_version_timestamp=prediction_version_timestamp,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in executor.map(
                upload_prediction, sample_id_to_prediction_singletons.items()
            ):
                if progress_bar is not None:
                    progress_bar.update(1)

    def create_or_update_prediction(
        self,
        sample_id: str,
        prediction_singletons: List[PredictionSingletonRepr],
        prediction_version_timestamp: int,
    ) -> None:
        prediction_singletons_for_sending = [
            singleton.to_dict() for singleton in prediction_singletons
        ]
        self._predictions_api.create_or_update_prediction_by_sample_id(
            body=prediction_singletons_for_sending,
            dataset_id=self.dataset_id,
            sample_id=sample_id,
            prediction_uuid_timestamp=prediction_version_timestamp,
        )
