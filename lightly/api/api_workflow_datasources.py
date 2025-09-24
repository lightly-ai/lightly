import time
import warnings
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import tqdm

from lightly.openapi_generated.swagger_client.models import (
    DatasourceConfig,
    DatasourceProcessedUntilTimestampRequest,
    DatasourceProcessedUntilTimestampResponse,
    DatasourcePurpose,
    DatasourceRawSamplesData,
)
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data_row import (
    DatasourceRawSamplesDataRow,
)


class _DatasourcesMixin:
    def download_raw_samples(
        self,
        from_: int = 0,
        to: Optional[int] = None,
        relevant_filenames_file_name: Optional[str] = None,
        use_redirected_read_url: bool = False,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Tuple[str, str]]:
        """Downloads filenames and read urls from the datasource.

        Only samples with timestamp between `from_` (inclusive) and `to` (inclusive)
        will be downloaded.

        Args:
            from_:
                Unix timestamp from which on samples are downloaded. Defaults to the
                very beginning (timestamp 0).
            to:
                Unix timestamp up to and including which samples are downloaded.
                Defaults to the current timestamp.
            relevant_filenames_file_name:
                Path to the relevant filenames text file in the cloud bucket.
                The path is relative to the datasource root. Optional.
            use_redirected_read_url:
                Flag for redirected read urls. When this flag is true,
                RedirectedReadUrls are returned instead of ReadUrls, meaning that the
                returned URLs have unlimited access to the file.
                Defaults to False. When S3DelegatedAccess is configured, this flag has
                no effect because RedirectedReadUrls are always returned.
            progress_bar:
                Tqdm progress bar to show how many samples have already been
                retrieved.

        Returns:
            A list of (filename, url) tuples where each tuple represents a sample.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.download_raw_samples()
            [('image-1.png', 'https://......'), ('image-2.png', 'https://......')]

        :meta private:  # Skip docstring generation
        """
        return self._download_raw_files(
            download_function=self._datasources_api.get_list_of_raw_samples_from_datasource_by_dataset_id,
            from_=from_,
            to=to,
            relevant_filenames_file_name=relevant_filenames_file_name,
            use_redirected_read_url=use_redirected_read_url,
            progress_bar=progress_bar,
        )

    def download_raw_predictions(
        self,
        task_name: str,
        from_: int = 0,
        to: Optional[int] = None,
        relevant_filenames_file_name: Optional[str] = None,
        run_id: Optional[str] = None,
        relevant_filenames_artifact_id: Optional[str] = None,
        use_redirected_read_url: bool = False,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Tuple[str, str]]:
        """Downloads all prediction filenames and read urls from the datasource.

        See `download_raw_predictions_iter` for details.

        :meta private:  # Skip docstring generation
        """
        return list(
            self.download_raw_predictions_iter(
                task_name=task_name,
                from_=from_,
                to=to,
                relevant_filenames_file_name=relevant_filenames_file_name,
                run_id=run_id,
                relevant_filenames_artifact_id=relevant_filenames_artifact_id,
                use_redirected_read_url=use_redirected_read_url,
                progress_bar=progress_bar,
            )
        )

    def download_raw_predictions_iter(
        self,
        task_name: str,
        from_: int = 0,
        to: Optional[int] = None,
        relevant_filenames_file_name: Optional[str] = None,
        run_id: Optional[str] = None,
        relevant_filenames_artifact_id: Optional[str] = None,
        use_redirected_read_url: bool = False,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> Iterator[Tuple[str, str]]:
        """Downloads prediction filenames and read urls from the datasource.

        Only samples with timestamp between `from_` (inclusive) and `to` (inclusive)
        will be downloaded.

        Args:
            task_name:
                Name of the prediction task.
            from_:
                Unix timestamp from which on samples are downloaded. Defaults to the
                very beginning (timestamp 0).
            to:
                Unix timestamp up to and including which samples are downloaded.
                Defaults to the current timestamp.
            relevant_filenames_file_name:
                Path to the relevant filenames text file in the cloud bucket.
                The path is relative to the datasource root. Optional.
            run_id:
                Run ID. Optional. Should be given along with
                `relevant_filenames_artifact_id` to download relevant files only.
            relevant_filenames_artifact_id:
                ID of the relevant filename artifact. Optional. Should be given along
                with `run_id` to download relevant files only. Note that this is
                different from `relevant_filenames_file_name`.
            use_redirected_read_url:
                Flag for redirected read urls. When this flag is true,
                RedirectedReadUrls are returned instead of ReadUrls, meaning that the
                returned URLs have unlimited access to the file.
                Defaults to False. When S3DelegatedAccess is configured, this flag has
                no effect because RedirectedReadUrls are always returned.
            progress_bar:
                Tqdm progress bar to show how many prediction files have already been
                retrieved.

        Returns:
            An iterator of (filename, url) tuples where each tuple represents a sample.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> task_name = "object-detection"
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> list(client.download_raw_predictions(task_name=task_name))
            [('.lightly/predictions/object-detection/image-1.json', 'https://......'),
             ('.lightly/predictions/object-detection/image-2.json', 'https://......')]

        :meta private:  # Skip docstring generation
        """
        if run_id is not None and relevant_filenames_artifact_id is None:
            raise ValueError(
                "'relevant_filenames_artifact_id' should not be `None` when 'run_id' "
                "is specified."
            )
        if run_id is None and relevant_filenames_artifact_id is not None:
            raise ValueError(
                "'run_id' should not be `None` when 'relevant_filenames_artifact_id' "
                "is specified."
            )
        relevant_filenames_kwargs = {}
        if run_id is not None and relevant_filenames_artifact_id is not None:
            relevant_filenames_kwargs["relevant_filenames_run_id"] = run_id
            relevant_filenames_kwargs[
                "relevant_filenames_artifact_id"
            ] = relevant_filenames_artifact_id

        yield from self._download_raw_files_iter(
            download_function=self._datasources_api.get_list_of_raw_samples_predictions_from_datasource_by_dataset_id,
            from_=from_,
            to=to,
            relevant_filenames_file_name=relevant_filenames_file_name,
            use_redirected_read_url=use_redirected_read_url,
            task_name=task_name,
            progress_bar=progress_bar,
            **relevant_filenames_kwargs,
        )

    def download_raw_metadata(
        self,
        from_: int = 0,
        to: Optional[int] = None,
        run_id: Optional[str] = None,
        relevant_filenames_artifact_id: Optional[str] = None,
        relevant_filenames_file_name: Optional[str] = None,
        use_redirected_read_url: bool = False,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Tuple[str, str]]:
        """Downloads all metadata filenames and read urls from the datasource.

        See `download_raw_metadata_iter` for details.

        :meta private:  # Skip docstring generation
        """
        return list(
            self.download_raw_metadata_iter(
                from_=from_,
                to=to,
                run_id=run_id,
                relevant_filenames_artifact_id=relevant_filenames_artifact_id,
                relevant_filenames_file_name=relevant_filenames_file_name,
                use_redirected_read_url=use_redirected_read_url,
                progress_bar=progress_bar,
            )
        )

    def download_raw_metadata_iter(
        self,
        from_: int = 0,
        to: Optional[int] = None,
        run_id: Optional[str] = None,
        relevant_filenames_artifact_id: Optional[str] = None,
        relevant_filenames_file_name: Optional[str] = None,
        use_redirected_read_url: bool = False,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> Iterator[Tuple[str, str]]:
        """Downloads all metadata filenames and read urls from the datasource.

        Only samples with timestamp between `from_` (inclusive) and `to` (inclusive)
        will be downloaded.

        Args:
            from_:
                Unix timestamp from which on samples are downloaded. Defaults to the
                very beginning (timestamp 0).
            to:
                Unix timestamp up to and including which samples are downloaded.
                Defaults to the current timestamp.
            relevant_filenames_file_name:
                Path to the relevant filenames text file in the cloud bucket.
                The path is relative to the datasource root. Optional.
            run_id:
                Run ID. Optional. Should be given along with
                `relevant_filenames_artifact_id` to download relevant files only.
            relevant_filenames_artifact_id:
                ID of the relevant filename artifact. Optional. Should be given along
                with `run_id` to download relevant files only. Note that this is
                different from `relevant_filenames_file_name`.
            use_redirected_read_url:
                Flag for redirected read urls. When this flag is true,
                RedirectedReadUrls are returned instead of ReadUrls, meaning that the
                returned URLs have unlimited access to the file.
                Defaults to False. When S3DelegatedAccess is configured, this flag has
                no effect because RedirectedReadUrls are always returned.
            progress_bar:
                Tqdm progress bar to show how many metadata files have already been
                retrieved.

        Returns:
            An iterator of (filename, url) tuples where each tuple represents a sample.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> list(client.download_raw_metadata_iter())
            [('.lightly/metadata/object-detection/image-1.json', 'https://......'),
             ('.lightly/metadata/object-detection/image-2.json', 'https://......')]

        :meta private:  # Skip docstring generation
        """
        if run_id is not None and relevant_filenames_artifact_id is None:
            raise ValueError(
                "'relevant_filenames_artifact_id' should not be `None` when 'run_id' "
                "is specified."
            )
        if run_id is None and relevant_filenames_artifact_id is not None:
            raise ValueError(
                "'run_id' should not be `None` when 'relevant_filenames_artifact_id' "
                "is specified."
            )
        relevant_filenames_kwargs = {}
        if run_id is not None and relevant_filenames_artifact_id is not None:
            relevant_filenames_kwargs["relevant_filenames_run_id"] = run_id
            relevant_filenames_kwargs[
                "relevant_filenames_artifact_id"
            ] = relevant_filenames_artifact_id

        yield from self._download_raw_files_iter(
            download_function=self._datasources_api.get_list_of_raw_samples_metadata_from_datasource_by_dataset_id,
            from_=from_,
            to=to,
            relevant_filenames_file_name=relevant_filenames_file_name,
            use_redirected_read_url=use_redirected_read_url,
            progress_bar=progress_bar,
            **relevant_filenames_kwargs,
        )

    def download_new_raw_samples(
        self,
        use_redirected_read_url: bool = False,
    ) -> List[Tuple[str, str]]:
        """Downloads filenames and read urls of unprocessed samples from the datasource.

        All samples after the timestamp of `ApiWorkflowClient.get_processed_until_timestamp()` are
        fetched. After downloading the samples, the timestamp is updated to the current time.
        This function can be repeatedly called to retrieve new samples from the datasource.

        Args:
            use_redirected_read_url:
                Flag for redirected read urls. When this flag is true,
                RedirectedReadUrls are returned instead of ReadUrls, meaning that the
                returned URLs have unlimited access to the file.
                Defaults to False. When S3DelegatedAccess is configured, this flag has
                no effect because RedirectedReadUrls are always returned.

        Returns:
            A list of (filename, url) tuples where each tuple represents a sample.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.download_new_raw_samples()
            [('image-3.png', 'https://......'), ('image-4.png', 'https://......')]
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
            use_redirected_read_url=use_redirected_read_url,
        )
        self.update_processed_until_timestamp(timestamp=to)
        return data

    def get_processed_until_timestamp(self) -> int:
        """Returns the timestamp until which samples have been processed.

        Returns:
            Unix timestamp of last processed sample.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.get_processed_until_timestamp()
            1684750513

        :meta private:  # Skip docstring generation
        """
        response: DatasourceProcessedUntilTimestampResponse = self._datasources_api.get_datasource_processed_until_timestamp_by_dataset_id(
            dataset_id=self.dataset_id
        )
        timestamp = int(response.processed_until_timestamp)
        return timestamp

    def update_processed_until_timestamp(self, timestamp: int) -> None:
        """Sets the timestamp until which samples have been processed.

        Args:
            timestamp:
                Unix timestamp of last processed sample.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset.
            >>> # All samples are processed at this moment.
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.download_new_raw_samples()
            []
            >>>
            >>> # Set timestamp to an earlier moment to reprocess samples
            >>> client.update_processed_until_timestamp(1684749813)
            >>> client.download_new_raw_samples()
            [('image-3.png', 'https://......'), ('image-4.png', 'https://......')]
        """
        body = DatasourceProcessedUntilTimestampRequest(
            processed_until_timestamp=timestamp
        )
        self._datasources_api.update_datasource_processed_until_timestamp_by_dataset_id(
            dataset_id=self.dataset_id,
            datasource_processed_until_timestamp_request=body,
        )

    def get_datasource(self) -> DatasourceConfig:
        """Returns the datasource of the current dataset.

        Returns:
            Datasource data of the datasource of the current dataset.

        Raises:
            ApiException if no datasource was configured.

        """
        return self._datasources_api.get_datasource_by_dataset_id(self.dataset_id)

    def set_azure_config(
        self,
        container_name: str,
        account_name: str,
        sas_token: str,
        thumbnail_suffix: Optional[
            str
        ] = ".lightly/thumbnails/[filename]_thumb.[extension]",
        purpose: str = DatasourcePurpose.INPUT_OUTPUT,
    ) -> None:
        """Sets the Azure configuration for the datasource of the current dataset.

        See our docs for a detailed explanation on how to setup Lightly with
        Azure: https://docs.lightly.ai/docs/azure

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

        """
        # TODO: Use DatasourceConfigAzure once we switch/update the api generator.
        self._datasources_api.update_datasource_by_dataset_id(
            datasource_config=DatasourceConfig.from_dict(
                {
                    "type": "AZURE",
                    "fullPath": container_name,
                    "thumbSuffix": thumbnail_suffix,
                    "accountName": account_name,
                    "accountKey": sas_token,
                    "purpose": purpose,
                }
            ),
            dataset_id=self.dataset_id,
        )

    def set_gcs_config(
        self,
        resource_path: str,
        project_id: str,
        credentials: str,
        thumbnail_suffix: Optional[
            str
        ] = ".lightly/thumbnails/[filename]_thumb.[extension]",
        purpose: str = DatasourcePurpose.INPUT_OUTPUT,
    ) -> None:
        """Sets the Google Cloud Storage configuration for the datasource of the
        current dataset.

        See our docs for a detailed explanation on how to setup Lightly with
        Google Cloud Storage: https://docs.lightly.ai/docs/google-cloud-storage

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

        """
        # TODO: Use DatasourceConfigGCS once we switch/update the api generator.
        self._datasources_api.update_datasource_by_dataset_id(
            datasource_config=DatasourceConfig.from_dict(
                {
                    "type": "GCS",
                    "fullPath": resource_path,
                    "thumbSuffix": thumbnail_suffix,
                    "gcsProjectId": project_id,
                    "gcsCredentials": credentials,
                    "purpose": purpose,
                }
            ),
            dataset_id=self.dataset_id,
        )

    def set_local_config(
        self,
        relative_path: str = "",
        web_server_location: Optional[str] = "http://localhost:3456",
        thumbnail_suffix: Optional[
            str
        ] = ".lightly/thumbnails/[filename]_thumb.[extension]",
        purpose: str = DatasourcePurpose.INPUT_OUTPUT,
    ) -> None:
        """Sets the local configuration for the datasource of the current dataset.

        Find a detailed explanation on how to setup Lightly with a local file
        server in our docs: https://docs.lightly.ai/getting_started/dataset_creation/dataset_creation_local_server.html

        Args:
            relative_path:
                Relative path from the mount root, for example: "path/to/my/data".
            web_server_location:
                Location of your local file server. Defaults to "http://localhost:3456".
            thumbnail_suffix:
                Where to save thumbnails of the images in the dataset, for
                example ".lightly/thumbnails/[filename]_thumb.[extension]".
                Set to None to disable thumbnails and use the full images from the
                datasource instead.
            purpose:
                Datasource purpose, determines if datasource is read only (INPUT)
                or can be written to as well (LIGHTLY, INPUT_OUTPUT).

        """
        # TODO: Use DatasourceConfigLocal once we switch/update the api generator.
        self._datasources_api.update_datasource_by_dataset_id(
            datasource_config=DatasourceConfig.from_dict(
                {
                    "type": "LOCAL",
                    "webServerLocation": web_server_location,
                    "fullPath": relative_path,
                    "thumbSuffix": thumbnail_suffix,
                    "purpose": purpose,
                }
            ),
            dataset_id=self.dataset_id,
        )

    def set_s3_config(
        self,
        resource_path: str,
        region: str,
        access_key: str,
        secret_access_key: str,
        thumbnail_suffix: Optional[
            str
        ] = ".lightly/thumbnails/[filename]_thumb.[extension]",
        purpose: str = DatasourcePurpose.INPUT_OUTPUT,
    ) -> None:
        """Sets the S3 configuration for the datasource of the current dataset.

        See our docs for a detailed explanation on how to setup Lightly with
        AWS S3: https://docs.lightly.ai/docs/aws-s3

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

        """
        # TODO: Use DatasourceConfigS3 once we switch/update the api generator.
        self._datasources_api.update_datasource_by_dataset_id(
            datasource_config=DatasourceConfig.from_dict(
                {
                    "type": "S3",
                    "fullPath": resource_path,
                    "thumbSuffix": thumbnail_suffix,
                    "s3Region": region,
                    "s3AccessKeyId": access_key,
                    "s3SecretAccessKey": secret_access_key,
                    "purpose": purpose,
                }
            ),
            dataset_id=self.dataset_id,
        )

    def set_s3_delegated_access_config(
        self,
        resource_path: str,
        region: str,
        role_arn: str,
        external_id: str,
        thumbnail_suffix: Optional[
            str
        ] = ".lightly/thumbnails/[filename]_thumb.[extension]",
        purpose: str = DatasourcePurpose.INPUT_OUTPUT,
    ) -> None:
        """Sets the S3 configuration for the datasource of the current dataset.

        See our docs for a detailed explanation on how to setup Lightly with
        AWS S3 and delegated access: https://docs.lightly.ai/docs/aws-s3#delegated-access

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

        """
        # TODO: Use DatasourceConfigS3 once we switch/update the api generator.
        self._datasources_api.update_datasource_by_dataset_id(
            datasource_config=DatasourceConfig.from_dict(
                {
                    "type": "S3DelegatedAccess",
                    "fullPath": resource_path,
                    "thumbSuffix": thumbnail_suffix,
                    "s3Region": region,
                    "s3ARN": role_arn,
                    "s3ExternalId": external_id,
                    "purpose": purpose,
                }
            ),
            dataset_id=self.dataset_id,
        )

    def set_obs_config(
        self,
        resource_path: str,
        obs_endpoint: str,
        obs_access_key_id: str,
        obs_secret_access_key: str,
        thumbnail_suffix: Optional[
            str
        ] = ".lightly/thumbnails/[filename]_thumb.[extension]",
        purpose: str = DatasourcePurpose.INPUT_OUTPUT,
    ) -> None:
        """Sets the Telekom OBS configuration for the datasource of the current dataset.

        Args:
            resource_path:
                OBS url of your dataset. For example, "obs://my_bucket/path/to/my/data".
            obs_endpoint:
                OBS endpoint.
            obs_access_key_id:
                OBS access key id.
            obs_secret_access_key:
                OBS secret access key.
            thumbnail_suffix:
                Where to save thumbnails of the images in the dataset, for
                example ".lightly/thumbnails/[filename]_thumb.[extension]".
                Set to None to disable thumbnails and use the full images from the
                datasource instead.
            purpose:
                Datasource purpose, determines if datasource is read only (INPUT)
                or can be written to as well (LIGHTLY, INPUT_OUTPUT).

        """
        # TODO: Use DatasourceConfigOBS once we switch/update the api generator.
        self._datasources_api.update_datasource_by_dataset_id(
            datasource_config=DatasourceConfig.from_dict(
                {
                    "type": "OBS",
                    "fullPath": resource_path,
                    "thumbSuffix": thumbnail_suffix,
                    "obsEndpoint": obs_endpoint,
                    "obsAccessKeyId": obs_access_key_id,
                    "obsSecretAccessKey": obs_secret_access_key,
                    "purpose": purpose,
                }
            ),
            dataset_id=self.dataset_id,
        )

    def get_prediction_read_url(
        self,
        filename: str,
    ) -> str:
        """Returns a read-url for .lightly/predictions/{filename}.

        Args:
            filename:
                Filename for which to get the read-url.

        Returns:
            A read-url to the file. Note that a URL will be returned even if the file does not
            exist.

        :meta private:  # Skip docstring generation
        """
        return self._datasources_api.get_prediction_file_read_url_from_datasource_by_dataset_id(
            dataset_id=self.dataset_id,
            file_name=filename,
        )

    def get_metadata_read_url(
        self,
        filename: str,
    ) -> str:
        """Returns a read-url for .lightly/metadata/{filename}.

        Args:
            filename:
                Filename for which to get the read-url.

        Returns:
            A read-url to the file. Note that a URL will be returned even if the file does not
            exist.

        :meta private:  # Skip docstring generation
        """
        return self._datasources_api.get_metadata_file_read_url_from_datasource_by_dataset_id(
            dataset_id=self.dataset_id,
            file_name=filename,
        )

    def get_custom_embedding_read_url(
        self,
        filename: str,
    ) -> str:
        """Returns a read-url for .lightly/embeddings/{filename}.

        Args:
            filename:
                Filename for which to get the read-url.

        Returns:
            A read-url to the file. Note that a URL will be returned even if the file does not
            exist.

        :meta private:  # Skip docstring generation
        """
        return self._datasources_api.get_custom_embedding_file_read_url_from_datasource_by_dataset_id(
            dataset_id=self.dataset_id,
            file_name=filename,
        )

    def list_datasource_permissions(
        self,
    ) -> Dict[str, Union[bool, Dict[str, str]]]:
        """Lists granted access permissions for the datasource set up with a dataset.

        Returns a string dictionary, with each permission mapped to a boolean value,
        see the example below. An additional ``errors`` key is present if any permission
        errors have been encountered. Permission errors are stored in a dictionary where
        permission names are keys and error messages are values.

        >>> from lightly.api import ApiWorkflowClient
        >>> client = ApiWorkflowClient(
        ...    token="MY_LIGHTLY_TOKEN", dataset_id="MY_DATASET_ID"
        ... )
        >>> client.list_datasource_permissions()
        {
            'can_read': True,
            'can_write': True,
            'can_list': False,
            'can_overwrite': True,
            'errors': {'can_list': 'error message'}
        }

        """
        return self._datasources_api.verify_datasource_by_dataset_id(
            dataset_id=self.dataset_id,
        ).to_dict()

    def _download_raw_files(
        self,
        download_function: Union[
            "DatasourcesApi.get_list_of_raw_samples_from_datasource_by_dataset_id",
            "DatasourcesApi.get_list_of_raw_samples_predictions_from_datasource_by_dataset_id",
            "DatasourcesApi.get_list_of_raw_samples_metadata_from_datasource_by_dataset_id",
        ],
        from_: int = 0,
        to: Optional[int] = None,
        relevant_filenames_file_name: Optional[str] = None,
        use_redirected_read_url: bool = False,
        progress_bar: Optional[tqdm.tqdm] = None,
        **kwargs,
    ) -> List[Tuple[str, str]]:
        return list(
            self._download_raw_files_iter(
                download_function=download_function,
                from_=from_,
                to=to,
                relevant_filenames_file_name=relevant_filenames_file_name,
                use_redirected_read_url=use_redirected_read_url,
                progress_bar=progress_bar,
                **kwargs,
            )
        )

    def _download_raw_files_iter(
        self,
        download_function: Union[
            "DatasourcesApi.get_list_of_raw_samples_from_datasource_by_dataset_id",
            "DatasourcesApi.get_list_of_raw_samples_predictions_from_datasource_by_dataset_id",
            "DatasourcesApi.get_list_of_raw_samples_metadata_from_datasource_by_dataset_id",
        ],
        from_: int = 0,
        to: Optional[int] = None,
        relevant_filenames_file_name: Optional[str] = None,
        use_redirected_read_url: bool = False,
        progress_bar: Optional[tqdm.tqdm] = None,
        **kwargs,
    ) -> Iterator[Tuple[str, str]]:
        if to is None:
            to = int(time.time())
        relevant_filenames_kwargs = (
            {"relevant_filenames_file_name": relevant_filenames_file_name}
            if relevant_filenames_file_name
            else dict()
        )

        listed_filenames = set()

        def get_samples(
            response: DatasourceRawSamplesData,
        ) -> Iterator[Tuple[str, str]]:
            for sample in response.data:
                if _sample_unseen_and_valid(
                    sample=sample,
                    relevant_filenames_file_name=relevant_filenames_file_name,
                    listed_filenames=listed_filenames,
                ):
                    listed_filenames.add(sample.file_name)
                    yield sample.file_name, sample.read_url
                if progress_bar is not None:
                    progress_bar.update(1)

        response: DatasourceRawSamplesData = download_function(
            dataset_id=self.dataset_id,
            var_from=from_,
            to=to,
            use_redirected_read_url=use_redirected_read_url,
            **relevant_filenames_kwargs,
            **kwargs,
        )
        yield from get_samples(response=response)
        while response.has_more:
            response: DatasourceRawSamplesData = download_function(
                dataset_id=self.dataset_id,
                cursor=response.cursor,
                use_redirected_read_url=use_redirected_read_url,
                **relevant_filenames_kwargs,
                **kwargs,
            )
            yield from get_samples(response=response)


def _sample_unseen_and_valid(
    sample: DatasourceRawSamplesDataRow,
    relevant_filenames_file_name: Optional[str],
    listed_filenames: Set[str],
) -> bool:
    # Note: We want to remove these checks eventually. Absolute paths and relative paths
    # with dot notation should be handled either in the API or the Worker. Duplicate
    # filenames should be handled in the Worker as handling it in the API would require
    # too much memory.
    if sample.file_name.startswith("/"):
        warnings.warn(
            UserWarning(
                f"Absolute file paths like {sample.file_name} are not supported"
                f" in relevant filenames file {relevant_filenames_file_name} due to blob storage"
            )
        )
        return False
    elif sample.file_name.startswith(("./", "../")):
        warnings.warn(
            UserWarning(
                f"Using dot notation ('./', '../') like in {sample.file_name} is not supported"
                f" in relevant filenames file {relevant_filenames_file_name} due to blob storage"
            )
        )
        return False
    elif sample.file_name in listed_filenames:
        warnings.warn(
            UserWarning(
                f"Duplicate filename {sample.file_name} in relevant"
                f" filenames file {relevant_filenames_file_name}"
            )
        )
        return False
    else:
        return True
