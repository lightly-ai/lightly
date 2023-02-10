import warnings
from typing import List, Optional

from lightly.api import utils
from lightly.openapi_generated.swagger_client import (CreateEntityResponse, DatasetCreateRequest, DatasetData, DatasetCreator, DatasetType, )
from lightly.openapi_generated.swagger_client.rest import ApiException


class _DatasetsMixin:
    @property
    def dataset_type(self) -> str:
        """Returns the dataset type of the current dataset."""
        dataset = self._get_current_dataset()
        return dataset.type  #  type: ignore

    def _get_current_dataset(self) -> DatasetData:
        """Returns the dataset with id == self.dataset_id."""
        return self.get_dataset_by_id(dataset_id=self.dataset_id)

    def dataset_exists(self, dataset_id: str) -> bool:
        """Returns True if a dataset with dataset_id exists."""
        try:
            self.get_dataset_by_id(dataset_id)
            return True
        except ApiException:
            return False

    def dataset_name_exists(
        self, dataset_name: str, shared: Optional[bool] = False
    ) -> bool:
        """Returns True if a dataset with dataset_name exists and False otherwise.

        Args:
            dataset_name:
                Name of the dataset.
            shared:
                If False, considers only datasets owned by the user.
                If True, considers only datasets which have been shared with the user.
                If None, considers all datasets the users has access to.
        """
        return bool(self.get_datasets_by_name(dataset_name=dataset_name, shared=shared))

    def get_dataset_by_id(self, dataset_id: str) -> DatasetData:
        """Returns the dataset for the given dataset id."""
        dataset: DatasetData = self._datasets_api.get_dataset_by_id(dataset_id)
        return dataset

    def get_datasets_by_name(
        self,
        dataset_name: str,
        shared: Optional[bool] = False,
    ) -> List[DatasetData]:
        """Returns datasets by name.

        An empty list is returned if no datasets with the name exist.

        Args:
            dataset_name:
                Name of the dataset.
            shared:
                If False, returns only datasets owned by the user. In this case at most
                one dataset will be returned.
                If True, returns only datasets which have been shared with the user. Can
                return multiple datasets.
                If None, returns datasets the users has access to. Can return multiple
                datasets.
        """
        datasets = []
        if not shared or shared is None:
            datasets.extend(
                self._datasets_api.get_datasets_query_by_name(
                    dataset_name=dataset_name,
                    exact=True,
                    shared=False,
                )
            )
        if shared or shared is None:
            datasets.extend(
                self._datasets_api.get_datasets_query_by_name(
                    dataset_name=dataset_name,
                    exact=True,
                    shared=True,
                )
            )
        return datasets

    def get_datasets(self, shared: Optional[bool] = False) -> List[DatasetData]:
        """Returns all datasets the user owns.

        Args:
            shared:
                If False, returns only datasets owned by the user.
                If True, returns only the datasets which have been shared with the user.
                If None, returns all datasets the user has access to (owned and shared).

        """
        datasets = []
        if not shared or shared is None:
            datasets.extend(
                utils.paginate_endpoint(
                    self._datasets_api.get_datasets,
                    shared=False,
                )
            )
        if shared or shared is None:
            datasets.extend(
                utils.paginate_endpoint(
                    self._datasets_api.get_datasets,
                    shared=True,
                )
            )
        return datasets

    def get_all_datasets(self) -> List[DatasetData]:
        """Returns all datasets the user has access to.

        DEPRECATED in favour of get_datasets(shared=None) and will be removed in the
        future.
        """
        warnings.warn(
            "get_all_datasets() is deprecated in favour of get_datasets(shared=None) "
            "and will be removed in the future.",
            PendingDeprecationWarning,
        )
        owned_datasets = self.get_datasets(shared=None)
        return owned_datasets

    def set_dataset_id_by_name(self, dataset_name: str, shared: Optional[bool] = False):
        """Sets the dataset id given the name of the dataset

        Args:
            dataset_name:
                The name of the dataset for which the dataset_id should be set as
                attribute.
            shared:
                If False, considers only datasets owned by the user.
                If True, considers only the datasets which have been shared with the user.
                If None, consider all datasets the user has access to (owned and shared).

        Raises: ValueError

        """
        datasets = self.get_datasets_by_name(dataset_name=dataset_name, shared=shared)
        if not datasets:
            raise ValueError(
                f"A dataset with the name '{dataset_name}' does not exist on the "
                f"Lightly Platform. Please create it first."
            )
        self._dataset_id = datasets[0].id
        if len(datasets) > 1:
            msg = (
                f"Found {len(datasets)} datasets with the name '{dataset_name}'. Their "
                f"ids are {[dataset.id for dataset in datasets]}. "
                f"The dataset_id of the client was set to '{self._dataset_id}'. "
            )
            if shared or shared is None:
                msg += (
                    f"We noticed that you set shared={shared} which also retrieves "
                    f"datasets shared with you. Set shared=False to only consider "
                    "datasets you own."
                )
            warnings.warn(msg)

    def create_dataset(
        self,
        dataset_name: str,
        dataset_type: str = DatasetType.IMAGES,
        dataset_creator: Optional[DatasetCreator] = DatasetCreator.USER_PIP,
    ):
        """Creates a dataset on the Lightly Platform.

        The dataset_id of the created dataset is stored in the client.dataset_id
        attribute and all further requests with the client will use the created dataset
        by default.

        Args:
            dataset_name:
                The name of the dataset to be created.
            dataset_type:
                The type of the dataset. We recommend to use the API provided constants
                `DatasetType.IMAGES` and `DatasetType.VIDEOS`.
            dataset_creator:
                Telling from where the dataset is created.

        Raises:
            ValueError: If a dataset with dataset_name already exists.

        Examples:
            >>> from lightly.api import ApiWorkflowClient
            >>> from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType
            >>>
            >>> client = lightly.api.ApiWorkflowClient(token="YOUR_TOKEN")
            >>> client.create_dataset('your-dataset-name', dataset_type=DatasetType.IMAGES)
            >>>
            >>> # or to work with videos
            >>> client.create_dataset('your-dataset-name', dataset_type=DatasetType.VIDEOS)
            >>>
            >>> # retrieving dataset_id of the created dataset
            >>> dataset_id = client.dataset_id
            >>>
            >>> # future client requests use the created dataset by default
            >>> client.dataset_type
            'Videos'
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
            dataset_creator=dataset_creator,
        )

    def _create_dataset_without_check_existing(
        self, dataset_name: str, dataset_type: str, dataset_creator: DatasetCreator
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
        body = DatasetCreateRequest(name=dataset_name, type=dataset_type, creator=dataset_creator)
        response: CreateEntityResponse = self._datasets_api.create_dataset(body=body)
        self._dataset_id = response.id

    def create_new_dataset_with_unique_name(
        self,
        dataset_basename: str,
        dataset_type: str = DatasetType.IMAGES,
        dataset_creator: Optional[DatasetCreator] = DatasetCreator.USER_PIP,
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
            dataset_creator:
                Telling from where the dataset is created.

        """
        if not self.dataset_name_exists(dataset_name=dataset_basename):
            self._create_dataset_without_check_existing(
                dataset_name=dataset_basename,
                dataset_type=dataset_type,
                dataset_creator=dataset_creator,
            )
        else:
            existing_datasets = self._datasets_api.get_datasets_query_by_name(
                dataset_name=dataset_basename,
                exact=False,
                shared=False,
            )
            existing_dataset_names = {dataset.name for dataset in existing_datasets}
            counter = 1
            dataset_name = f"{dataset_basename}_{counter}"
            while dataset_name in existing_dataset_names:
                counter += 1
                dataset_name = f"{dataset_basename}_{counter}"
            self._create_dataset_without_check_existing(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                dataset_creator=dataset_creator,
            )

    def delete_dataset_by_id(self, dataset_id: str):
        """Deletes a dataset on the Lightly Platform.

        Args:
            dataset_id:
                The id of the dataset to be deleted.

        """
        self._datasets_api.delete_dataset_by_id(dataset_id=dataset_id)
        del self._dataset_id
