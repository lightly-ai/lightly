import time
from typing import List, Optional, Tuple

from lightly.openapi_generated.swagger_client.models.datasource_processed_until_timestamp_request import DatasourceProcessedUntilTimestampRequest
from lightly.openapi_generated.swagger_client.models.datasource_processed_until_timestamp_response import DatasourceProcessedUntilTimestampResponse

from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data import DatasourceRawSamplesData



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

    def update_s3_config(
        self,
        resource_path: str,
        thumbnail_suffix: Optional[str],
        region: str,
        access_key: str,
        secret_access_key: str,
    ) -> None:
        """Sets the S3 configuration for the datasource of the current dataset.
        
        Args:
            resource_path:
                S3 url of your dataset, for example "s3://my_bucket/path/to/dataset".
            thumbnail_suffix:
                Where to save thumbnails of the images in the dataset, for
                example ".lightly/thumbnails/[filename]_thumb.[extension]". 
                Set to None to disable thumbnails and use the full images from the 
                datasource instead.
            region:
                S3 region where the dataset bucket is located, for example
                "eu-central-1".
            access_key:
                S3 access key.
            secret_access_key:
                Secret for the S3 access key.

        """
        # Cannot use the DatasourceConfigS3 model because it does not include
        # the type, fullPath and thumbSuffix arguments. This is a limitation
        # of swagger-codegen.
        # TODO: Use DatasourceConfigS3 once we switch/update the generator.
        self._datasources_api.update_datasource_by_dataset_id(
            body={
                'type': 'S3',
                'fullPath': resource_path,
                'thumbSuffix': thumbnail_suffix,
                's3Region': region,
                's3AccessKeyId': access_key,
                's3SecretAccessKey': secret_access_key,
            },
            dataset_id=self.dataset_id,
        )
