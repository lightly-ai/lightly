from typing import Dict, Optional, Union

from lightly.openapi_generated.swagger_client.models import (
    DatasourceConfig,
    DatasourcePurpose,
)


class _DatasourcesMixin:
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
