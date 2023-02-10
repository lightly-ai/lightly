from pytest_mock import MockerFixture

from lightly.active_learning.config.selection_config import SelectionConfig
from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import (
    TagData,
    DatasetsApi,
    DatasetType,
    DatasetCreator,
    DatasetCreateRequest,
)
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


class TestApiWorkflowDatasets(MockedApiWorkflowSetup):
    def setUp(self, token="token_xyz", dataset_id="dataset_id_xyz") -> None:
        super().setUp(token, dataset_id)
        self.api_workflow_client._datasets_api.reset()

    def test_create_dataset_new(self):
        self.api_workflow_client.create_dataset(dataset_name="dataset_new")
        assert isinstance(self.api_workflow_client.dataset_id, str)
        assert len(self.api_workflow_client.dataset_id) > 0

    def test_create_dataset_existing(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.create_dataset(dataset_name="dataset_1")

    def test_dataset_name_exists__own_not_existing(self):
        assert not self.api_workflow_client.dataset_name_exists(
            dataset_name="not_existing_dataset"
        )

    def test_dataset_name_exists__own_existing(self):
        assert self.api_workflow_client.dataset_name_exists(dataset_name="dataset_1")

    def test_dataset_name_exists__shared_existing(self):
        assert self.api_workflow_client.dataset_name_exists(
            dataset_name="shared_dataset_1", shared=True
        )

    def test_dataset_name_exists__shared_not_existing(self):
        assert not self.api_workflow_client.dataset_name_exists(
            dataset_name="not_existing_dataset", shared=True
        )

    def test_dataset_name_exists__own_and_shared_existing(self):
        assert self.api_workflow_client.dataset_name_exists(
            dataset_name="dataset_1", shared=None
        )
        assert self.api_workflow_client.dataset_name_exists(
            dataset_name="shared_dataset_1", shared=None
        )

    def test_dataset_name_exists__own_and_shared_not_existing(self):
        assert not self.api_workflow_client.dataset_name_exists(
            dataset_name="not_existing_dataset", shared=None
        )

    def test_get_datasets_by_name__own_not_existing(self):
        datasets = self.api_workflow_client.get_datasets_by_name(
            dataset_name="shared_dataset_1", shared=False
        )
        assert datasets == []

    def test_get_datasets_by_name__own_existing(self):
        datasets = self.api_workflow_client.get_datasets_by_name(
            dataset_name="dataset_1", shared=False
        )
        assert all(dataset.name == "dataset_1" for dataset in datasets)
        assert len(datasets) == 1

    def test_get_datasets_by_name__shared_not_existing(self):
        datasets = self.api_workflow_client.get_datasets_by_name(
            dataset_name="dataset_1", shared=True
        )
        assert datasets == []

    def test_get_datasets_by_name__shared_existing(self):
        datasets = self.api_workflow_client.get_datasets_by_name(
            dataset_name="shared_dataset_1", shared=True
        )
        assert all(dataset.name == "shared_dataset_1" for dataset in datasets)
        assert len(datasets) == 1

    def test_get_datasets_by_name__own_and_shared_not_existing(self):
        datasets = self.api_workflow_client.get_datasets_by_name(
            dataset_name="not_existing_dataset", shared=None
        )
        assert datasets == []

    def test_get_datasets_by_name__own_and_shared_existing(self):
        datasets = self.api_workflow_client.get_datasets_by_name(
            dataset_name="dataset_1", shared=None
        )
        assert all(dataset.name == "dataset_1" for dataset in datasets)
        assert len(datasets) == 1

        datasets = self.api_workflow_client.get_datasets_by_name(
            dataset_name="shared_dataset_1", shared=True
        )
        assert all(dataset.name == "shared_dataset_1" for dataset in datasets)
        assert len(datasets) == 1

    def test_get_all_datasets(self):
        datasets = self.api_workflow_client.get_all_datasets()
        dataset_names = {dataset.name for dataset in datasets}
        assert "dataset_1" in dataset_names
        assert "shared_dataset_1" in dataset_names

    def test_create_dataset_with_counter(self):
        self.api_workflow_client.create_dataset(dataset_name="basename")
        n_tries = 3
        for i in range(n_tries):
            self.api_workflow_client.create_new_dataset_with_unique_name(
                dataset_basename="basename"
            )
        assert (
            self.api_workflow_client._datasets_api.datasets[-1].name
            == f"basename_{n_tries}"
        )

    def test_create_dataset_with_counter_nonexisting(self):
        self.api_workflow_client.create_dataset(dataset_name="basename")
        self.api_workflow_client.create_new_dataset_with_unique_name(
            dataset_basename="baseName"
        )
        assert self.api_workflow_client._datasets_api.datasets[-1].name == "baseName"

    def test_set_dataset_id__own_success(self):
        self.api_workflow_client.set_dataset_id_by_name("dataset_1", shared=False)
        assert self.api_workflow_client.dataset_id == "dataset_1_id"

    def test_set_dataset_id__own_error(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.set_dataset_id_by_name(
                "shared_dataset_1", shared=False
            )

    def test_set_dataset_id__shared_success(self):
        self.api_workflow_client.set_dataset_id_by_name("shared_dataset_1", shared=True)
        assert self.api_workflow_client.dataset_id == "shared_dataset_1_id"

    def test_set_dataset_id__shared_error(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.set_dataset_id_by_name("dataset_1", shared=True)

    def test_set_dataset_id__own_and_shared_success(self):
        self.api_workflow_client.set_dataset_id_by_name("dataset_1", shared=None)
        assert self.api_workflow_client.dataset_id == "dataset_1_id"

        self.api_workflow_client.set_dataset_id_by_name("shared_dataset_1", shared=None)
        assert self.api_workflow_client.dataset_id == "shared_dataset_1_id"

    def test_set_dataset_id__own_and_shared_error(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.set_dataset_id_by_name(
                "not_existing_dataset", shared=None
            )

    def test_delete_dataset(self):
        self.api_workflow_client.create_dataset(dataset_name="dataset_to_delete")
        self.api_workflow_client.delete_dataset_by_id(
            self.api_workflow_client.dataset_id
        )
        assert not hasattr(self, "_dataset_id")

    def test_dataset_type(self):
        self.api_workflow_client.create_dataset(dataset_name="some_dataset")
        assert self.api_workflow_client.dataset_type == "Images"

    def test_get_datasets(self):
        num_datasets_before = len(self.api_workflow_client.get_datasets())
        self.api_workflow_client.create_new_dataset_with_unique_name("dataset")
        num_datasets_after = len(self.api_workflow_client.get_datasets())
        assert num_datasets_before + 1 == num_datasets_after


def test_create_dataset__dataset_creator_unset(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    client = ApiWorkflowClient()
    client._datasets_api = mocker.create_autospec(DatasetsApi)

    client.create_dataset(dataset_name="name")
    expected_body = DatasetCreateRequest(
        name="name", type=DatasetType.IMAGES, creator=DatasetCreator.USER_PIP
    )
    client._datasets_api.create_dataset.assert_called_once_with(expected_body)


def test_create_dataset__dataset_creator_set(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    client = ApiWorkflowClient()
    client._datasets_api = mocker.create_autospec(DatasetsApi)

    client.create_dataset(
        dataset_name="name", dataset_creator=DatasetCreator.USER_WORKER
    )
    expected_body = DatasetCreateRequest(
        name="name", type=DatasetType.IMAGES, creator=DatasetCreator.USER_WORKER
    )
    client._datasets_api.create_dataset.assert_called_once_with(expected_body)


def test_create_new_dataset_with_unique_name__dataset_creator_unset(
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    client = ApiWorkflowClient()
    client._datasets_api = mocker.create_autospec(DatasetsApi)

    client.create_new_dataset_with_unique_name(dataset_basename="name")

    expected_body = DatasetCreateRequest(
        name="name", type=DatasetType.IMAGES, creator=DatasetCreator.USER_PIP
    )
    client._datasets_api.create_dataset.assert_called_once_with(expected_body)


def test_create_new_dataset_with_unique_name__dataset_creator_set(
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    client = ApiWorkflowClient()
    client._datasets_api = mocker.create_autospec(DatasetsApi)

    client.create_new_dataset_with_unique_name(
        dataset_basename="name", dataset_creator=DatasetCreator.USER_WORKER
    )

    expected_body = DatasetCreateRequest(
        name="name", type=DatasetType.IMAGES, creator=DatasetCreator.USER_WORKER
    )
    client._datasets_api.create_dataset.assert_called_once_with(expected_body)
