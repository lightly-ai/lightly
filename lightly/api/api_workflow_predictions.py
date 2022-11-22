from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union, Optional

import tqdm

from lightly.api.prediction_singletons import PredictionSingletonClassificationRepr
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
                The uuid timestamp in nanoseconds since 1970. Get it via 'int(time.time())'.

        Example:
          >>> import time
          >>> from lightly.api import ApiWorkflowClient
          >>> from lightly.openapi_generated.swagger_client import (
          >>>     PredictionTaskSchema,
          >>>     TaskType,
          >>>     PredictionTaskSchemaCategory,
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
          >>> client = ApiWorkflowClient(
          >>>     token="MY_LIGHTLY_TOKEN", dataset_id="MY_DATASET_ID"
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
        filename_to_prediction_singletons: Union[
            Dict[str, List[PredictionSingletonClassificationRepr]],
            Dict[str, List[PredictionSingletonObjectDetection]],
            Dict[str, List[PredictionSingletonInstanceSegmentation]],
            Dict[str, List[PredictionSingletonKeypointDetection]],
        ],
        prediction_uuid_timestamp: int,
        progress_bar: Optional[tqdm.tqdm] = None,
        max_workers: int = 8

    ) -> None:
        """Creates or updates the predictions for specific samples

        Args:
            filename_to_prediction_singletons
                A mapping from the filename of the sample to its corresponding prediction singletons.
                All singletons must be of the same type.

            prediction_uuid_timestamp:
                The uuid timestamp in nanoseconds since 1970. Get it via 'int(time.time())'.

            progress_bar:
                Tqdm progress bar to show how many prediction files have already been
                uploaded.

        Example:


        """

        samples = retry(
            self._samples_api.get_samples_partial_by_dataset_id,
            dataset_id=self.dataset_id,
            mode=SamplePartialMode.FILENAMES,
        )
        filename_to_sample_id = {sample.file_name: sample.id for sample in samples}

        def upload_prediction(filename_and_predictions_tuple):
            (filename, predictions) = filename_and_predictions_tuple
            sample_id = filename_to_sample_id[filename]
            prediction_singletons_for_sending = [vars(singleton) for singleton in predictions]
            self._predictions_api.create_or_update_prediction_by_sample_id(
                body=prediction_singletons_for_sending,
                dataset_id=self.dataset_id,
                sample_id=sample_id,
                prediction_uuid_timestamp=prediction_uuid_timestamp,
            )

        # handle the case where len(sample_ids) < max_workers
        max_workers = min(len(filename_to_sample_id), max_workers)
        max_workers = max(max_workers, 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in executor.map(upload_prediction, filename_to_prediction_singletons.items()):
                if progress_bar is not None:
                    progress_bar.update(1)
