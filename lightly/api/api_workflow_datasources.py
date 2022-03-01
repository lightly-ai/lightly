import time
import requests

from typing import List, Tuple, Dict

from lightly.openapi_generated.swagger_client.models.datasource_processed_until_timestamp_request import DatasourceProcessedUntilTimestampRequest
from lightly.openapi_generated.swagger_client.models.datasource_processed_until_timestamp_response import DatasourceProcessedUntilTimestampResponse

from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data import DatasourceRawSamplesData
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_predictions_data import DatasourceRawSamplesPredictionsData



class _DatasourcesMixin:

    def download_raw_samples(
        self, from_: int = 0, to: int = None
    ) -> List[Tuple[str, str]]:
        """Downloads all filenames and read urls from the datasource between `from_` and `to`.

        Samples which have timestamp == `from_` or timestamp == `to` will also be included.
        
        Args:
            from_: 
                Unix timestamp from which on samples are downloaded.
            to: 
                Unix timestamp up to and including which samples are downloaded.
        
        Returns:
           A list of (filename, url) tuples, where each tuple represents a sample

        """
        if to is None:
            to = int(time.time())
        response: DatasourceRawSamplesData = self._datasources_api.get_list_of_raw_samples_from_datasource_by_dataset_id(
            dataset_id=self.dataset_id,
            _from=from_,
            to=to,
        )
        cursor = response.cursor
        samples = response.data
        while response.has_more:
            response: DatasourceRawSamplesData = self._datasources_api.get_list_of_raw_samples_from_datasource_by_dataset_id(
                dataset_id=self.dataset_id, cursor=cursor
            )
            cursor = response.cursor
            samples.extend(response.data)
        samples = [(s.file_name, s.read_url) for s in samples]
        return samples

    def download_new_raw_samples(self) -> List[Tuple[str, str]]:
        """Downloads filenames and read urls of unprocessed samples from the datasource.

        All samples after the timestamp of `ApiWorkflowClient.get_processed_until_timestamp()` are 
        fetched. After downloading the samples the timestamp is updated to the current time.
        This function can be repeatedly called to retrieve new samples from the datasource.
        
        Returns:
            A list of (filename, url) tuples, where each tuple represents a sample

        """
        from_ = self.get_processed_until_timestamp()
        
        if from_ != 0:
            # We already processed samples at some point.
            # Add 1 because the samples with timestamp == from_
            # have already been processed
            from_ += 1

        to = int(time.time())
        data = self.download_raw_samples(from_=from_, to=to)
        self.update_processed_until_timestamp(timestamp=to)
        return data

    def get_processed_until_timestamp(self) -> int:
        """Returns the timestamp until which samples have been processed.
        
        Returns:
            Unix timestamp of last processed sample
        """
        response: DatasourceProcessedUntilTimestampResponse = (
            self._datasources_api.get_datasource_processed_until_timestamp_by_dataset_id(
                dataset_id=self.dataset_id
            )
        )
        timestamp = int(response.processed_until_timestamp)
        return timestamp

    def update_processed_until_timestamp(self, timestamp: int) -> None:
        """Sets the timestamp until which samples have been processed.
        
        Args:
            timestamp: 
                Unix timestamp of last processed sample
        """
        body = DatasourceProcessedUntilTimestampRequest(
            processed_until_timestamp=timestamp
        )
        self._datasources_api.update_datasource_processed_until_timestamp_by_dataset_id(
            dataset_id=self.dataset_id, body=body
        )

    def get_datasource(self):
        """Calls the api to return the datasource of the current dataset.

        Returns:
            Datasource data of the datasource of the current dataset.

        Raises:
            ApiException if no datasource was configured.

        """
        return self._datasources_api.get_datasource_by_dataset_id(
            self.dataset_id
        )

    def download_prediction_task_names(
        self,
        tasks_filename: str = 'tasks.json',
    ) -> List[str]:
        """Downloads the task names stored in the tasks.json file.

        Args:
            tasks_filename:
                Naming convention of the tasks file (always tasks.json).

        Returns a list of task names.
        
        """
        tasks_read_url = self._datasources_api.get_prediction_file_read_url_from_datasource_by_dataset_id(
            self.dataset_id,
            tasks_filename,
        )

        response = requests.get(tasks_read_url, stream=True)
        response.raise_for_status()

        tasks = response.json()
        if isinstance(tasks, dict):
            task_names = tasks.keys()
        elif isinstance(tasks, List):
            task_names = tasks
        else:
            raise ValueError('Invalid task format for tasks.json!')


        if any([not isinstance(task_name, str) for task_name in task_names]):
            raise ValueError('Task names in task.json must be strings!')

        return task_names

    def download_task_schema(
        self,
        task_name: str,
        schema_filename: str = 'schema.json',
    ) -> Tuple[str, Dict]:
        """Downloads the schema of a given prediction task.

        Args:
            task_name:
                Name of the prediction task.
            schema_filename:
                Naming convention of the schema files (always schema.json).

        Returns the task description (e.g. classification, object-detection)
        and the schema of the specified task.
        
        """
        schema_read_url = self._datasources_api.get_prediction_file_read_url_from_datasource_by_dataset_id(
            self.dataset_id,
            f'{task_name}/{schema_filename}',
        )
        response = requests.get(schema_read_url, stream=True)
        response.raise_for_status()

        schema = response.json()
        task_description = schema.get('task_description', None)
        return task_description, schema

    def download_raw_predictions(
        self,
        task_name: str,
        from_: int = 0,
        to: int = None
    ) -> List[Tuple[str, str]]:
        """Downloads all prediction filenames and read urls from the datasource between `from_` and `to`.

        Samples which have timestamp == `from_` or timestamp == `to` will also be included.
        
        Args:
            task_name:
                Name of the prediction task.
            from_: 
                Unix timestamp from which on samples are downloaded.
            to: 
                Unix timestamp up to and including which samples are downloaded.
        
        Returns:
           A list of (filename, url) tuples, where each tuple represents a sample

        """
        if to is None:
            to = int(time.time())
        response: DatasourceRawSamplesPredictionsData = self._datasources_api.get_list_of_raw_samples_predictions_from_datasource_by_dataset_id(
            self.dataset_id,
            task_name,
            _from=from_,
            to=to,
        )
        cursor = response.cursor
        samples = response.data
        while response.has_more:
            response: DatasourceRawSamplesPredictionsData = self._datasources_api.get_list_of_raw_samples_predictions_from_datasource_by_dataset_id(
                self.dataset_id,
                task_name,
                cursor=cursor
            )
            cursor = response.cursor
            samples.extend(response.data)
        samples = [(s.file_name, s.read_url) for s in samples]
        return samples
