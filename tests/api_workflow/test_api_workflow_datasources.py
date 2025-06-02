from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client.models import (
    DatasourceConfigAzure,
    DatasourceConfigGCS,
    DatasourceConfigLOCAL,
    DatasourceConfigS3,
    DatasourceConfigS3DelegatedAccess,
    DatasourcePurpose,
)
from lightly.openapi_generated.swagger_client.models.datasource_config_verify_data import (
    DatasourceConfigVerifyData,
)
from lightly.openapi_generated.swagger_client.models.datasource_config_verify_data_errors import (
    DatasourceConfigVerifyDataErrors,
)


class TestDatasourcesMixin:
    def test_set_azure_config(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "update_datasource_by_dataset_id",
        )
        client.set_azure_config(
            container_name="my-container/name",
            account_name="my-account-name",
            sas_token="my-sas-token",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
        )
        kwargs = client._datasources_api.update_datasource_by_dataset_id.call_args[1]
        assert isinstance(
            kwargs["datasource_config"].actual_instance, DatasourceConfigAzure
        )

    def test_set_gcs_config(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "update_datasource_by_dataset_id",
        )
        client.set_gcs_config(
            resource_path="gs://my-bucket/my-dataset",
            project_id="my-project-id",
            credentials="my-credentials",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
        )
        kwargs = client._datasources_api.update_datasource_by_dataset_id.call_args[1]
        assert isinstance(
            kwargs["datasource_config"].actual_instance, DatasourceConfigGCS
        )

    def test_set_local_config(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "update_datasource_by_dataset_id",
        )
        client.set_local_config(
            web_server_location="http://localhost:1234",
            relative_path="path/to/my/data",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
            purpose=DatasourcePurpose.INPUT,
        )
        kwargs = client._datasources_api.update_datasource_by_dataset_id.call_args[1]
        datasource_config = kwargs["datasource_config"].actual_instance
        assert isinstance(datasource_config, DatasourceConfigLOCAL)
        assert datasource_config.type == "LOCAL"
        assert datasource_config.web_server_location == "http://localhost:1234"
        assert datasource_config.full_path == "path/to/my/data"
        assert (
            datasource_config.thumb_suffix
            == ".lightly/thumbnails/[filename]-thumb-[extension]"
        )
        assert datasource_config.purpose == DatasourcePurpose.INPUT

        # Test defaults
        client.set_local_config()
        kwargs = client._datasources_api.update_datasource_by_dataset_id.call_args[1]
        datasource_config = kwargs["datasource_config"].actual_instance
        assert isinstance(datasource_config, DatasourceConfigLOCAL)
        assert datasource_config.type == "LOCAL"
        assert datasource_config.web_server_location == "http://localhost:3456"
        assert datasource_config.full_path == ""
        assert (
            datasource_config.thumb_suffix
            == ".lightly/thumbnails/[filename]_thumb.[extension]"
        )
        assert datasource_config.purpose == DatasourcePurpose.INPUT_OUTPUT

    def test_set_s3_config(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "update_datasource_by_dataset_id",
        )
        client.set_s3_config(
            resource_path="s3://my-bucket/my-dataset",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
            region="eu-central-1",
            access_key="my-access-key",
            secret_access_key="my-secret-access-key",
        )
        kwargs = client._datasources_api.update_datasource_by_dataset_id.call_args[1]
        assert isinstance(
            kwargs["datasource_config"].actual_instance, DatasourceConfigS3
        )

    def test_set_s3_delegated_access_config(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "update_datasource_by_dataset_id",
        )
        client.set_s3_delegated_access_config(
            resource_path="s3://my-bucket/my-dataset",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
            region="eu-central-1",
            role_arn="arn:aws:iam::000000000000:role.test",
            external_id="my-external-id",
        )
        kwargs = client._datasources_api.update_datasource_by_dataset_id.call_args[1]
        assert isinstance(
            kwargs["datasource_config"].actual_instance,
            DatasourceConfigS3DelegatedAccess,
        )

    def test_list_datasource_permissions(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        client._datasources_api.verify_datasource_by_dataset_id = mocker.MagicMock(
            return_value=DatasourceConfigVerifyData(
                canRead=True,
                canWrite=True,
                canList=False,
                canOverwrite=True,
                errors=None,
            ),
        )
        assert client.list_datasource_permissions() == {
            "can_read": True,
            "can_write": True,
            "can_list": False,
            "can_overwrite": True,
        }

    def test_list_datasource_permissions__error(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        client._datasources_api.verify_datasource_by_dataset_id = mocker.MagicMock(
            return_value=DatasourceConfigVerifyData(
                canRead=True,
                canWrite=True,
                canList=False,
                canOverwrite=True,
                errors=DatasourceConfigVerifyDataErrors(
                    canRead=None,
                    canWrite=None,
                    canList="error message",
                    canOverwrite=None,
                ),
            ),
        )
        assert client.list_datasource_permissions() == {
            "can_read": True,
            "can_write": True,
            "can_list": False,
            "can_overwrite": True,
            "errors": {
                "can_list": "error message",
            },
        }
