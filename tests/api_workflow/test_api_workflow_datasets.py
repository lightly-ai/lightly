from typing import List

import pytest
from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient, api_workflow_datasets
from lightly.openapi_generated.swagger_client.api import DatasetsApi
from lightly.openapi_generated.swagger_client.models import (
    Creator,
    DatasetCreateRequest,
    DatasetData,
    DatasetType,
)
from lightly.openapi_generated.swagger_client.rest import ApiException
from tests.api_workflow import utils
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


def _get_datasets(count: int) -> List[DatasetData]:
    return [
        DatasetData(
            name=f"mock_dataset_{i}",
            id=utils.generate_id(),
            last_modified_at=0,
            type=DatasetType.IMAGES,
            img_type="full",
            size_in_bytes=-1,
            n_samples=-1,
            created_at=0,
            user_id="user_0",
        )
        for i in range(count)
    ]


class TestApiWorkflowDatasets(MockedApiWorkflowSetup):
    def setUp(self, token="token_xyz", dataset_id="dataset_id_xyz") -> None:
        super().setUp(token, dataset_id)
        self.api_workflow_client._datasets_api.reset()

    def test_create_dataset_existing(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.create_dataset(dataset_name="dataset_1")

    def test_dataset_name_exists__own_not_existing(self):
        assert not self.api_workflow_client.dataset_name_exists(
            dataset_name="not_existing_dataset"
        )

    def test_dataset_exists__raises_error(self):
        with self.assertRaises(ApiException) as e:
            self.api_workflow_client.dataset_exists(dataset_id=None)
            assert e.status != 404

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


def test_create_new_dataset_with_unique_name__new_name(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(ApiWorkflowClient, "dataset_name_exists", return_value=False)
    mocked_create_dataset = mocker.patch.object(
        ApiWorkflowClient, "_create_dataset_without_check_existing"
    )
    dataset_name = "dataset-name"
    dataset_type = DatasetType.IMAGES
    client = ApiWorkflowClient()
    client.create_new_dataset_with_unique_name(
        dataset_basename=dataset_name, dataset_type=dataset_type
    )
    mocked_create_dataset.assert_called_once_with(
        dataset_name=dataset_name,
        dataset_type=dataset_type,
    )


def test_create_new_dataset_with_unique_name__name_exists(
    mocker: MockerFixture,
) -> None:
    datasets = _get_datasets(1)
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(ApiWorkflowClient, "dataset_name_exists", return_value=True)
    mocked_create_dataset = mocker.patch.object(
        ApiWorkflowClient, "_create_dataset_without_check_existing"
    )
    mocked_datasets_api = mocker.MagicMock()
    dataset_name = datasets[0].name
    dataset_type = datasets[0].type
    actual_dataset_name = f"{dataset_name}_1"
    client = ApiWorkflowClient()
    client._datasets_api = mocked_datasets_api
    client.create_new_dataset_with_unique_name(
        dataset_basename=dataset_name, dataset_type=dataset_type
    )
    mocked_datasets_api.get_datasets_query_by_name.assert_called_once_with(
        dataset_name=dataset_name,
        exact=False,
        shared=False,
        page_offset=0,
        page_size=5000,
    )
    mocked_create_dataset.assert_called_once_with(
        dataset_name=actual_dataset_name,
        dataset_type=dataset_type,
    )


def test_dataset_exists(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_get_dataset = mocker.patch.object(ApiWorkflowClient, "get_dataset_by_id")
    dataset_id = "dataset-id"
    client = ApiWorkflowClient()
    assert client.dataset_exists(dataset_id)
    mocked_get_dataset.assert_called_once_with(dataset_id)


def test_dataset_exists__not_found(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient, "get_dataset_by_id", side_effect=ApiException(status=404)
    )
    client = ApiWorkflowClient()
    assert not client.dataset_exists("foo")


def test_dataset_exists__error(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient, "get_dataset_by_id", side_effect=RuntimeError("some error")
    )
    client = ApiWorkflowClient()
    with pytest.raises(RuntimeError) as exception:
        client.dataset_exists("foo")
        assert str(exception.value) == "some error"


def test_dataset_type(mocker: MockerFixture) -> None:
    dataset = _get_datasets(1)[0]
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(ApiWorkflowClient, "_get_current_dataset", return_value=dataset)
    client = ApiWorkflowClient()
    assert client.dataset_type == dataset.type


def test_delete_dataset(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mock_datasets_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._dataset_id = "foo"
    client._datasets_api = mock_datasets_api
    client.delete_dataset_by_id("foobar")
    mock_datasets_api.delete_dataset_by_id.assert_called_once_with(dataset_id="foobar")
    assert not hasattr(client, "_dataset_id")


def test_get_datasets__shared(mocker: MockerFixture) -> None:
    datasets = _get_datasets(2)
    # Returns the same set of datasets twice. API client should remove duplicates
    mocked_pagination = mocker.patch.object(
        api_workflow_datasets.utils,
        "paginate_endpoint",
        side_effect=[datasets, datasets],
    )
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mock_datasets_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._datasets_api = mock_datasets_api
    datasets = client.get_datasets(shared=True)
    unique_dataset_ids = set([dataset.id for dataset in datasets])
    assert len(unique_dataset_ids) == len(datasets)

    assert mocked_pagination.call_count == 2
    call_args = mocked_pagination.call_args_list
    assert call_args[0][0] == (mock_datasets_api.get_datasets,)
    assert call_args[0][1] == {"shared": True}
    assert call_args[1][0] == (mock_datasets_api.get_datasets,)
    assert call_args[1][1] == {"get_assets_of_team": True}


def test_get_datasets__not_shared(mocker: MockerFixture) -> None:
    mocked_pagination = mocker.patch.object(
        api_workflow_datasets.utils, "paginate_endpoint"
    )
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mock_datasets_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._datasets_api = mock_datasets_api
    client.get_datasets(shared=False)
    mocked_pagination.assert_called_once_with(
        mock_datasets_api.get_datasets, shared=False
    )


def test_get_datasets__shared_None(mocker: MockerFixture) -> None:
    mocked_pagination = mocker.patch.object(
        api_workflow_datasets.utils, "paginate_endpoint"
    )
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mock_datasets_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._datasets_api = mock_datasets_api
    client.get_datasets(shared=None)
    assert mocked_pagination.call_count == 3


def test_get_datasets_by_name__not_shared__paginated(mocker: MockerFixture) -> None:
    datasets = _get_datasets(3)
    # Returns the same set of datasets twice. API client should remove duplicates
    mocker.patch.object(
        api_workflow_datasets.utils,
        "paginate_endpoint",
        return_value=iter([datasets[0], datasets[1]]),
    )
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mock_datasets_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._datasets_api = mock_datasets_api

    datasets_not_shared = client.get_datasets_by_name(
        shared=False, dataset_name="mocked_dataset_0"
    )


def test_get_datasets_by_name__not_shared__paginated(mocker: MockerFixture) -> None:
    datasets = _get_datasets(3)
    # Returns the same set of datasets twice. API client should remove duplicates.
    mocker.patch.object(
        api_workflow_datasets.utils,
        "paginate_endpoint",
        # There's one call to paginate_endpoint.
        # It returns a paginated list of datasets.
        return_value=iter([datasets[0], datasets[1]]),
    )
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mock_datasets_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._datasets_api = mock_datasets_api

    # The dataset names are created in the _get_datasets function and correspond
    # to mocked_dataset_{index}.
    datasets_not_shared = client.get_datasets_by_name(
        shared=False, dataset_name="mocked_dataset_0"
    )
    assert len(datasets_not_shared) == 2


def test_get_datasets_by_name__shared__paginated(mocker: MockerFixture) -> None:
    datasets = _get_datasets(3)
    # Returns the same set of datasets twice. API client should remove duplicates.
    mocker.patch.object(
        api_workflow_datasets.utils,
        "paginate_endpoint",
        side_effect=[
            # There are two calls to paginate_endpoint to get all the team's datasets.
            iter([datasets[2]]),
            iter([]),
        ],
    )
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mock_datasets_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._datasets_api = mock_datasets_api

    # The dataset names are created in the _get_datasets function and correspond
    # to mocked_dataset_{index}.
    datasets_shared = client.get_datasets_by_name(
        shared=True, dataset_name="mocked_dataset_2"
    )
    assert len(datasets_shared) == 1


def test_get_datasets_by_name__shared_None__paginated(mocker: MockerFixture) -> None:
    datasets = _get_datasets(3)
    # Returns the same set of datasets twice. API client should remove duplicates.
    mocker.patch.object(
        api_workflow_datasets.utils,
        "paginate_endpoint",
        side_effect=[
            # There are three calls to paginate_endpoint. The first call
            # gets all the user's datasets. The second and third calls get
            # all the team's datasets.
            # The first call returns a paginated list of datasets.
            iter([datasets[0], datasets[1]]),
            iter([datasets[2]]),
            iter([]),
        ],
    )
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mock_datasets_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._datasets_api = mock_datasets_api

    # The dataset names are created in the _get_datasets function and correspond
    # to mocked_dataset_{index}.
    datasets_shared_none = client.get_datasets_by_name(
        shared=None, dataset_name="mocked_dataset_0"
    )
    assert len(datasets_shared_none) == 3


def test_set_dataset_id__error(mocker: MockerFixture):
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(ApiWorkflowClient, "get_datasets_by_name", return_value=[])
    client = ApiWorkflowClient()
    with pytest.raises(ValueError) as exception:
        client.set_dataset_id_by_name("dataset_1")
    assert str(exception.value) == (
        "A dataset with the name 'dataset_1' does not exist on the "
        "Lightly Platform. Please create it first."
    )


def test_set_dataset_id__warning_not_shared(mocker: MockerFixture) -> None:
    datasets = _get_datasets(2)
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient, "get_datasets_by_name", return_value=datasets
    )
    mocked_warn = mocker.patch("warnings.warn")
    client = ApiWorkflowClient()

    dataset_name = datasets[0].name
    dataset_id = datasets[0].id
    client.set_dataset_id_by_name(dataset_name, shared=False)
    assert client.dataset_id == dataset_id
    mocked_warn.assert_called_once_with(
        f"Found 2 datasets with the name '{dataset_name}'. Their "
        f"ids are {[dataset.id for dataset in datasets]}. "
        f"The dataset_id of the client was set to '{dataset_id}'. "
    )


def test_set_dataset_id__warning_shared(mocker: MockerFixture) -> None:
    datasets = _get_datasets(2)
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient, "get_datasets_by_name", return_value=datasets
    )
    mocked_warn = mocker.patch("warnings.warn")
    client = ApiWorkflowClient()

    dataset_name = datasets[0].name
    dataset_id = datasets[0].id
    client.set_dataset_id_by_name(dataset_name, shared=True)
    assert client.dataset_id == dataset_id
    mocked_warn.assert_called_once_with(
        f"Found 2 datasets with the name '{dataset_name}'. Their "
        f"ids are {[dataset.id for dataset in datasets]}. "
        f"The dataset_id of the client was set to '{dataset_id}'. "
        "We noticed that you set shared=True which also retrieves "
        "datasets shared with you. Set shared=False to only consider "
        "datasets you own."
    )


def test_set_dataset_id__success(mocker: MockerFixture) -> None:
    datasets = _get_datasets(1)
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient, "get_datasets_by_name", return_value=datasets
    )
    client = ApiWorkflowClient()
    client.set_dataset_id_by_name(datasets[0].name)
    assert client.dataset_id == datasets[0].id


def test_create_dataset(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    client = ApiWorkflowClient()
    client._creator = Creator.USER_PIP
    client._datasets_api = mocker.create_autospec(DatasetsApi)

    client.create_dataset(dataset_name="name")
    expected_body = DatasetCreateRequest(
        name="name", type=DatasetType.IMAGES, creator=Creator.USER_PIP
    )
    client._datasets_api.create_dataset.assert_called_once_with(
        dataset_create_request=expected_body
    )


def test_create_dataset__error(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(ApiWorkflowClient, "dataset_name_exists", return_value=True)

    client = ApiWorkflowClient()
    with pytest.raises(ValueError) as exception:
        client.create_dataset(dataset_name="name")
        assert str(exception.value) == (
            "A dataset with the name 'name' already exists! Please use "
            "the `set_dataset_id_by_name()` method instead if you intend to reuse "
            "an existing dataset."
        )
