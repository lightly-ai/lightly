from typing import List, Optional

from lightly.openapi_generated.swagger_client.models.create_entity_response import CreateEntityResponse
from lightly.openapi_generated.swagger_client.models.dataset_create_request import DatasetCreateRequest

from lightly.openapi_generated.swagger_client.models.dataset_data import DatasetData
from lightly.openapi_generated.swagger_client.rest import ApiException

class _DatasetsMixin:

    @property
    def dataset_type(self) -> str:
        """Returns the dataset type of the current dataset.

        """
        dataset = self._get_current_dataset()
        return dataset.type # type: ignore

    def _get_current_dataset(self) -> DatasetData:
        """Returns the dataset with id == self.dataset_id.

        """
        return self.get_dataset_by_id(self.dataset_id)

    def dataset_exists(self, dataset_id: str) -> bool:
        """Returns True if a dataset with dataset_id exists. """
        try:
            self.get_dataset_by_id(dataset_id)
            return True
        except ApiException:
            return False

    def dataset_name_exists(self, dataset_name: str) -> bool:
        """Returns True if a dataset with dataset_name exists."""
        return any(dataset.name == dataset_name for dataset in self.get_all_datasets())

    def get_dataset_by_id(self, dataset_id: str) -> DatasetData:
        """Returns the dataset for the given dataset id. """
        dataset: DatasetData = self._datasets_api.get_dataset_by_id(dataset_id)
        return dataset

    def get_datasets(self, shared: bool = False) -> List[DatasetData]:
        """Returns all datasets the user owns.

        Args:
            shared:
                If True, only returns the datasets which have been shared with
                the user.

        """
        datasets = self._datasets_api.get_datasets(shared=shared)
        return datasets

    def get_all_datasets(self) -> List[DatasetData]:
        """Returns all datasets the user has access to. """
        owned_datasets = self.get_datasets(shared=False)
        shared_datasets = self.get_datasets(shared=True)
        owned_datasets.extend(shared_datasets)
        return owned_datasets

    def set_dataset_id_by_name(self, dataset_name: str):
        """Sets the dataset id given the name of the dataset

        Args:
            dataset_name:
                The name of the dataset for which the dataset_id
                should be set as attribute

        Raises: ValueError

        """
        current_datasets: List[DatasetData] = self.get_all_datasets()

        try:
            dataset_with_specified_name = next(dataset for dataset in current_datasets if dataset.name == dataset_name)
            self._dataset_id = dataset_with_specified_name.id
        except StopIteration:
            raise ValueError(
                f"A dataset with the name {dataset_name} does not exist on the "
                f"Lightly Platform. Please create it first."
            )

    def create_dataset(self, dataset_name: str, dataset_type: Optional[str] = None):
        """Creates a dataset on the Lightly Platform..

        Raises a ValueError if a dataset with the given name already exists.

        Args:
            dataset_name:
                The name of the dataset to be created.
            dataset_type:
                The type of the dataset. We recommend to use the API provided constants
                `DatasetType.IMAGES` and `DatasetType.VIDEOS`.

        Examples:
            >>> from lightly.api import ApiWorkflowClient
            >>> from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType
            >>>
            >>> client = lightly.api.ApiWorkflowClient(token="YOUR_TOKEN")
            >>> client.create_dataset('your-dataset-name', dataset_type=DatasetType.IMAGES)
            >>>
            >>> # or to work with videos
            >>> client.create_dataset('your-dataset-name', dataset_type=DatasetType.VIDEOS)
        """

        if self.dataset_name_exists(dataset_name=dataset_name):
            raise ValueError(
                f"A dataset with the name '{dataset_name}' already exists! Please use "
                f"the `set_dataset_id_by_name()` method instead if you intend to reuse "
                f"an existing dataset."
            )
        self._create_dataset_without_check_existing(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
        )


    def _create_dataset_without_check_existing(
        self,
        dataset_name: str,
        dataset_type: Optional[str] = None,
    ):
        """Creates a dataset on the Lightly Platform.

        No checking if a dataset with such a name already exists is performed.

        Args:
            dataset_name:
                The name of the dataset to be created.
            dataset_type:
                The type of the dataset. We recommend to use the API provided
                constants `DatasetType.IMAGES` and `DatasetType.VIDEOS`.

        """
        body = DatasetCreateRequest(name=dataset_name, type=dataset_type)
        response: CreateEntityResponse = self._datasets_api.create_dataset(body=body)
        self._dataset_id = response.id

    def create_new_dataset_with_unique_name(
        self,
        dataset_basename: str,
        dataset_type: Optional[str] = None,
    ):
        """Creates a new dataset on the Lightly Platform.

        If a dataset with the specified name already exists,
        a counter is added to the name to be able to still create it.

        Args:
            dataset_basename:
                The name of the dataset to be created.
            dataset_type:
                The type of the dataset. We recommend to use the API provided
                constants `DatasetType.IMAGES` and `DatasetType.VIDEOS`.

        """
        current_datasets = self.get_datasets()
        current_datasets_names = [dataset.name for dataset in current_datasets]

        if dataset_basename not in current_datasets_names:
            self._create_dataset_without_check_existing(
                dataset_name=dataset_basename,
                dataset_type=dataset_type
            )
        else:
            counter = 1
            dataset_name = f"{dataset_basename}_{counter}"
            while dataset_name in current_datasets_names:
                counter += 1
                dataset_name = f"{dataset_basename}_{counter}"
            self._create_dataset_without_check_existing(
                dataset_name=dataset_name,
                dataset_type=dataset_type
            )

    def delete_dataset_by_id(self, dataset_id: str):
        """Deletes a dataset on the Lightly Platform.

        Args:
            dataset_id:
                The id of the dataset to be deleted.

        """
        self._datasets_api.delete_dataset_by_id(dataset_id=dataset_id)
        del self._dataset_id
