import time
from typing import Dict, List, Optional, Tuple, Union

import tqdm

from lightly.openapi_generated.swagger_client.model.datasource_config import DatasourceConfig
from lightly.openapi_generated.swagger_client.model.datasource_processed_until_timestamp_request import DatasourceProcessedUntilTimestampRequest
from lightly.openapi_generated.swagger_client.model.datasource_purpose import DatasourcePurpose
from lightly.openapi_generated.swagger_client.model.datasource_raw_samples_data import DatasourceRawSamplesData


class _DatasourcesMixin:

    def _download_raw_files(
            self,
            download_function: Union[
                "DatasourcesApi.get_list_of_raw_samples_from_datasource_by_dataset_id",
                "DatasourcesApi.get_list_of_raw_samples_predictions_from_datasource_by_dataset_id",
                "DatasourcesApi.get_list_of_raw_samples_metadata_from_datasource_by_dataset_id"
            ],
            from_: int = 0,
            to: Optional[int] = None,
            relevant_filenames_file_name: Optional[str] = None,
            use_redirected_read_url: Optional[bool] = False,
            progress_bar: Optional[tqdm.tqdm] = None,
            **kwargs
    ):
        if to is None:
            to = int(time.time())
        relevant_filenames_kwargs = {
            "relevant_filenames_file_name": relevant_filenames_file_name
        } if relevant_filenames_file_name else dict()

        response: DatasourceRawSamplesData = download_function(
            dataset_id=self.dataset_id,
            _from=from_,
            to=to,
            use_redirected_read_url=use_redirected_read_url,
            **relevant_filenames_kwargs,
            **kwargs
        )
        cursor = response.cursor
        samples = response.data
        if progress_bar is not None:
            progress_bar.update(len(response.data))
        while response.has_more:
            response: DatasourceRawSamplesData = download_function(
                dataset_id=self.dataset_id,
                cursor=cursor,
                use_redirected_read_url=use_redirected_read_url,
                **relevant_filenames_kwargs,
                **kwargs
            )
            cursor = response.cursor
            samples.extend(response.data)
            if progress_bar is not None:
                progress_bar.update(len(response.data))
        samples = [(s.file_name, s.read_url) for s in samples]
        return samples

    def download_raw_samples(
            self,
            from_: int = 0,
            to: Optional[int] = None,
            relevant_filenames_file_name: Optional[str] = None,
            use_redirected_read_url: Optional[bool] = False,
            progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Tuple[str, str]]:
        """Downloads all filenames and read urls from the datasource between `from_` and `to`.

        Samples which have timestamp == `from_` or timestamp == `to` will also be included.
        
        Args:
            from_: 
                Unix timestamp from which on samples are downloaded.
            to: 
                Unix timestamp up to and including which samples are downloaded.
            relevant_filenames_file_name:
                The path to the relevant filenames text file in the cloud bucket.
                The path is relative to the datasource root.
            use_redirected_read_url:
                By default this is set to false unless a S3DelegatedAccess is configured in which
                case its always true and this param has no effect.
                When true this will return RedirectedReadUrls instead of ReadUrls meaning that 
                returned URLs allow for unlimited access to the file
            progress_bar:
                Tqdm progress bar to show how many samples have already been
                retrieved.
        
        Returns:
           A list of (filename, url) tuples, where each tuple represents a sample

        """
        samples = self._download_raw_files(
            download_function=self._datasources_api.get_list_of_raw_samples_from_datasource_by_dataset_id,
            from_=from_,
            to=to,
            relevant_filenames_file_name=relevant_filenames_file_name,
            use_redirected_read_url=use_redirected_read_url,
            progress_bar=progress_bar,
        )
        return samples

    def download_raw_predictions(
            self,
            task_name: str,
            from_: int = 0,
            to: Optional[int] = None,
            relevant_filenames_file_name: Optional[str] = None,
            use_redirected_read_url: Optional[bool] = False,
            progress_bar: Optional[tqdm.tqdm] = None,
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
            relevant_filenames_file_name:
                The path to the relevant filenames text file in the cloud bucket.
                The path is relative to the datasource root.
            use_redirected_read_url:
                By default this is set to false unless a S3DelegatedAccess is configured in which
                case its always true and this param has no effect.
                When true this will return RedirectedReadUrls instead of ReadUrls meaning that 
                returned URLs allow for unlimited access to the file
            progress_bar:
                Tqdm progress bar to show how many prediction files have already been
                retrieved.

        Returns:
           A list of (filename, url) tuples, where each tuple represents a sample

        """
        samples = self._download_raw_files(
            download_function=self._datasources_api.get_list_of_raw_samples_predictions_from_datasource_by_dataset_id,
            from_=from_,
            to=to,
            relevant_filenames_file_name=relevant_filenames_file_name,
            use_redirected_read_url=use_redirected_read_url,
            task_name=task_name,
            progress_bar=progress_bar,
        )
        return samples

    def download_raw_metadata(
            self,
            from_: int = 0,
            to: Optional[int] = None,
            relevant_filenames_file_name: Optional[str] = None,
            use_redirected_read_url: Optional[bool] = False,
            progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Tuple[str, str]]:
        """Downloads all metadata filenames and read urls from the datasource between `from_` and `to`.

        Samples which have timestamp == `from_` or timestamp == `to` will also be included.

        Args:
            from_:
                Unix timestamp from which on samples are downloaded.
            to:
                Unix timestamp up to and including which samples are downloaded.
            relevant_filenames_file_name:
                The path to the relevant filenames text file in the cloud bucket.
                The path is relative to the datasource root.
            use_redirected_read_url:
                By default this is set to false unless a S3DelegatedAccess is configured in which
                case its always true and this param has no effect.
                When true this will return RedirectedReadUrls instead of ReadUrls meaning that 
                returned URLs allow for unlimited access to the file
            progress_bar:
                Tqdm progress bar to show how many metadata files have already been
                retrieved.

        Returns:
           A list of (filename, url) tuples, where each tuple represents a sample

        """
        samples = self._download_raw_files(
            self._datasources_api.get_list_of_raw_samples_metadata_from_datasource_by_dataset_id,
            from_=from_,
            to=to,
            relevant_filenames_file_name=relevant_filenames_file_name,
            use_redirected_read_url=use_redirected_read_url,
            progress_bar=progress_bar,
        )
        return samples

    def download_new_raw_samples(
        self,
        use_redirected_read_url: Optional[bool] = False,
    ) -> List[Tuple[str, str]]:
        """Downloads filenames and read urls of unprocessed samples from the datasource.

        All samples after the timestamp of `ApiWorkflowClient.get_processed_until_timestamp()` are 
        fetched. After downloading the samples the timestamp is updated to the current time.
        This function can be repeatedly called to retrieve new samples from the datasource.

        Args:
            use_redirected_read_url:
                By default this is set to false unless a S3DelegatedAccess is configured in which
                case its always true and this param has no effect.
                When true this will return RedirectedReadUrls instead of ReadUrls meaning that 
                returned URLs allow for unlimited access to the file
        
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
        data = self.download_raw_samples(
            from_=from_,
            to=to,
            relevant_filenames_file_name=None,
            use_redirected_read_url=use_redirected_read_url
        )
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

    def get_datasource(self) -> DatasourceConfig:
        """Calls the api to return the datasource of the current dataset.

        Returns:
            Datasource data of the datasource of the current dataset.

        Raises:
            ApiException if no datasource was configured.

        """
        return self._datasources_api.get_datasource_by_dataset_id(
            self.dataset_id
        )

    def set_azure_config(
        self,
        container_name: str,
        account_name: str,
        sas_token: str,
        thumbnail_suffix: Optional[str] = ".lightly/thumbnails/[filename]_thumb.[extension]",
        purpose: str = DatasourcePurpose.INPUT_OUTPUT,
    ) -> None:
        """Sets the Azure configuration for the datasource of the current dataset.
        
        Find a detailed explanation on how to setup Lightly with 
        Azure Blob Storage in our docs: https://docs.lightly.ai/getting_started/dataset_creation/dataset_creation_azure_storage.html#

        Args:
            container_name:
                Container name of the dataset, for example: "my-container/path/to/my/data".
            account_name:
                Azure account name.
            sas_token:
                Secure Access Signature token.
            thumbnail_suffix:
                Where to save thumbnails of the images in the dataset, for
                example ".lightly/thumbnails/[filename]_thumb.[extension]". 
                Set to None to disable thumbnails and use the full images from the 
                datasource instead.
            purpose:
                Datasource purpose, determines if datasource is read only (INPUT)
                or can be written to as well (LIGHTLY, INPUT_OUTPUT).
                The latter is required when Lightly extracts frames from input videos.

        """
        # TODO: Use DatasourceConfigAzure once we switch/update the api generator.
        self._datasources_api.update_datasource_by_dataset_id(
            body={
                'type': 'AZURE',
                'fullPath': container_name,
                'thumbSuffix': thumbnail_suffix,
                'accountName': account_name,
                'accountKey': sas_token,
                'purpose': purpose,
            },
            dataset_id=self.dataset_id,
        )

    def set_gcs_config(
        self,
        resource_path: str,
        project_id: str,
        credentials: str,
        thumbnail_suffix: Optional[str] = ".lightly/thumbnails/[filename]_thumb.[extension]",
        purpose: str = DatasourcePurpose.INPUT_OUTPUT,
    ) -> None:
        """Sets the Google Cloud Storage configuration for the datasource of the
        current dataset.

        Find a detailed explanation on how to setup Lightly with 
        Google Cloud Storage in our docs: https://docs.lightly.ai/getting_started/dataset_creation/dataset_creation_gcloud_bucket.html
        
        Args:
            resource_path:
                GCS url of your dataset, for example: "gs://my_bucket/path/to/my/data"
            project_id:
                GCS project id.
            credentials:
                Content of the credentials JSON file stringified which you 
                download from Google Cloud Platform.
            thumbnail_suffix:
                Where to save thumbnails of the images in the dataset, for
                example ".lightly/thumbnails/[filename]_thumb.[extension]". 
                Set to None to disable thumbnails and use the full images from the 
                datasource instead.
            purpose:
                Datasource purpose, determines if datasource is read only (INPUT)
                or can be written to as well (LIGHTLY, INPUT_OUTPUT).
                The latter is required when Lightly extracts frames from input videos.

        """
        # TODO: Use DatasourceConfigGCS once we switch/update the api generator.
        self._datasources_api.update_datasource_by_dataset_id(
            body={
                'type': 'GCS',
                'fullPath': resource_path,
                'thumbSuffix': thumbnail_suffix,
                'gcsProjectId': project_id,
                'gcsCredentials': credentials,
                'purpose': purpose,
            },
            dataset_id=self.dataset_id,
        )

    def set_local_config(
        self,
        resource_path: str,
        thumbnail_suffix: Optional[str] = ".lightly/thumbnails/[filename]_thumb.[extension]",
    ) -> None:
        """Sets the local configuration for the datasource of the current dataset.

        Find a detailed explanation on how to setup Lightly with a local file
        server in our docs: https://docs.lightly.ai/getting_started/dataset_creation/dataset_creation_local_server.html
        
        Args:
            resource_path:
                Url to your local file server, for example: "http://localhost:1234/path/to/my/data".
            thumbnail_suffix:
                Where to save thumbnails of the images in the dataset, for
                example ".lightly/thumbnails/[filename]_thumb.[extension]". 
                Set to None to disable thumbnails and use the full images from the 
                datasource instead.
        """
        # TODO: Use DatasourceConfigLocal once we switch/update the api generator.
        self._datasources_api.update_datasource_by_dataset_id(
            body={
                'type': 'LOCAL',
                'fullPath': resource_path,
                'thumbSuffix': thumbnail_suffix,
                'purpose': DatasourcePurpose.INPUT_OUTPUT,
            },
            dataset_id=self.dataset_id,
        )

    def set_s3_config(
        self,
        resource_path: str,
        region: str,
        access_key: str,
        secret_access_key: str,
        thumbnail_suffix: Optional[str] = ".lightly/thumbnails/[filename]_thumb.[extension]",
        purpose: str = DatasourcePurpose.INPUT_OUTPUT,
    ) -> None:
        """Sets the S3 configuration for the datasource of the current dataset.
        
        Args:
            resource_path:
                S3 url of your dataset, for example "s3://my_bucket/path/to/my/data".
            region:
                S3 region where the dataset bucket is located, for example "eu-central-1".
            access_key:
                S3 access key.
            secret_access_key:
                Secret for the S3 access key.
            thumbnail_suffix:
                Where to save thumbnails of the images in the dataset, for
                example ".lightly/thumbnails/[filename]_thumb.[extension]". 
                Set to None to disable thumbnails and use the full images from the 
                datasource instead.
            purpose:
                Datasource purpose, determines if datasource is read only (INPUT)
                or can be written to as well (LIGHTLY, INPUT_OUTPUT).
                The latter is required when Lightly extracts frames from input videos.

        """
        # TODO: Use DatasourceConfigS3 once we switch/update the api generator.
        self._datasources_api.update_datasource_by_dataset_id(
            body={
                'type': 'S3',
                'fullPath': resource_path,
                'thumbSuffix': thumbnail_suffix,
                's3Region': region,
                's3AccessKeyId': access_key,
                's3SecretAccessKey': secret_access_key,
                'purpose': purpose,
            },
            dataset_id=self.dataset_id,
        )

    def set_s3_delegated_access_config(
        self,
        resource_path: str,
        region: str,
        role_arn: str,
        external_id: str,
        thumbnail_suffix: Optional[str] = ".lightly/thumbnails/[filename]_thumb.[extension]",
        purpose: str = DatasourcePurpose.INPUT_OUTPUT,
    ) -> None:
        """Sets the S3 configuration for the datasource of the current dataset.
        
        Args:
            resource_path:
                S3 url of your dataset, for example "s3://my_bucket/path/to/my/data".
            region:
                S3 region where the dataset bucket is located, for example "eu-central-1".
            role_arn:
                Unique ARN identifier of the role.
            external_id:
                External ID of the role.
            thumbnail_suffix:
                Where to save thumbnails of the images in the dataset, for
                example ".lightly/thumbnails/[filename]_thumb.[extension]". 
                Set to None to disable thumbnails and use the full images from the 
                datasource instead.
            purpose:
                Datasource purpose, determines if datasource is read only (INPUT)
                or can be written to as well (LIGHTLY, INPUT_OUTPUT).
                The latter is required when Lightly extracts frames from input videos.

        """
        # TODO: Use DatasourceConfigS3 once we switch/update the api generator.
        self._datasources_api.update_datasource_by_dataset_id(
            body={
                'type': 'S3DelegatedAccess',
                'fullPath': resource_path,
                'thumbSuffix': thumbnail_suffix,
                's3Region': region,
                's3ARN': role_arn,
                's3ExternalId': external_id,
                'purpose': purpose,
            },
            dataset_id=self.dataset_id,
        )


    def get_prediction_read_url(
        self,
        filename: str,
    ):
        """Returns a read-url for .lightly/predictions/{filename}.
    
        Args:
            filename:
                Filename for which to get the read-url.

        Returns the read-url. If the file does not exist, a read-url is returned
        anyways.
        
        """
        return self._datasources_api.get_prediction_file_read_url_from_datasource_by_dataset_id(
            self.dataset_id,
            filename,
        )

    def get_metadata_read_url(
        self,
        filename: str,
    ):
        """Returns a read-url for .lightly/metadata/{filename}.
    
        Args:
            filename:
                Filename for which to get the read-url.

        Returns the read-url. If the file does not exist, a read-url is returned
        anyways.
        
        """
        return self._datasources_api.get_metadata_file_read_url_from_datasource_by_dataset_id(
            self.dataset_id,
            filename,
        )
