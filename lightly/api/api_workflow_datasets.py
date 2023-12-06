import warnings
from itertools import chain
from typing import Iterator, List, Optional, Set

from lightly.api import utils
from lightly.openapi_generated.swagger_client.models import (
    CreateEntityResponse,
    DatasetCreateRequest,
    DatasetData,
    DatasetType,
)
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
        """Checks if a dataset exists.

        Args:
            dataset_id: Dataset ID.

        Returns:
            True if the dataset exists and False otherwise.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> client.create_dataset("your-dataset-name", dataset_type=DatasetType.IMAGES)
            >>> dataset_id = client.dataset_id
            >>> client.dataset_exists(dataset_id=dataset_id)
            True
        """
        try:
            self.get_dataset_by_id(dataset_id)
            return True
        except ApiException as exception:
            if exception.status == 404:  # Not Found
                return False
            raise exception

    def dataset_name_exists(
        self, dataset_name: str, shared: Optional[bool] = False
    ) -> bool:
        """Checks if a dataset with the given name exists.

        There can be multiple datasets with the same name accessible to the current
        user. This can happen if either:
        * A dataset has been explicitly shared with the user
        * The user has access to team datasets
        The `shared` flag controls whether these datasets are checked.

        Args:
            dataset_name:
                Name of the dataset.
            shared:
                * If False (default), checks only datasets owned by the user.
                * If True, checks datasets which have been shared with the user,
                including team datasets. Excludes user's own datasets.
                * If None, checks all datasets the users has access to.

        Returns:
            A boolean value indicating whether any dataset with the given name exists.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> client.create_dataset("your-dataset-name", dataset_type=DatasetType.IMAGES)
            >>> client.dataset_name_exists(dataset_name="your-dataset-name")
            True
        """
        return bool(self.get_datasets_by_name(dataset_name=dataset_name, shared=shared))

    def get_dataset_by_id(self, dataset_id: str) -> DatasetData:
        """Fetches a dataset by ID.

        Args:
            dataset_id: Dataset ID.

        Returns:
            The dataset with the given dataset id.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> client.create_dataset("your-dataset-name", dataset_type=DatasetType.IMAGES)
            >>> dataset_id = client.dataset_id
            >>> client.get_dataset_by_id(dataset_id=dataset_id)
            {'created_at': 1685009504596,
             'datasource_processed_until_timestamp': 1685009513,
             'datasources': ['646f346004d77b4e1424e67e', '646f346004d77b4e1424e695'],
             'id': '646f34608a5613b57d8b73c9',
             'img_type': 'full',
             'type': 'Images',
             ...}
        """
        dataset: DatasetData = self._datasets_api.get_dataset_by_id(dataset_id)
        return dataset

    def get_datasets_by_name(
        self,
        dataset_name: str,
        shared: Optional[bool] = False,
    ) -> List[DatasetData]:
        """Fetches datasets by name.

        There can be multiple datasets with the same name accessible to the current
        user. This can happen if either:
        * A dataset has been explicitly shared with the user
        * The user has access to team datasets
        The `shared` flag controls whether these datasets are returned.

        Args:
            dataset_name:
                Name of the target dataset.
            shared:
                * If False (default), returns only datasets owned by the user. In this
                case at most one dataset will be returned.
                * If True, returns datasets which have been shared with the user,
                including team datasets. Excludes user's own datasets. Can return
                multiple datasets.
                * If None, returns all datasets the users has access to. Can return
                multiple datasets.

        Returns:
            A list of datasets that match the name. If no datasets with the name exist,
            an empty list is returned.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> client.create_dataset("your-dataset-name", dataset_type=DatasetType.IMAGES)
            >>> client.get_datasets_by_name(dataset_name="your-dataset-name")
            [{'created_at': 1685009504596,
             'datasource_processed_until_timestamp': 1685009513,
             'datasources': ['646f346004d77b4e1424e67e', '646f346004d77b4e1424e695'],
             'id': '646f34608a5613b57d8b73c9',
             'img_type': 'full',
             'type': 'Images',
             ...}]
            >>>
            >>> # Non-existent dataset
            >>> client.get_datasets_by_name(dataset_name="random-name")
            []
        """
        datasets: List[DatasetData] = []
        if not shared or shared is None:
            datasets.extend(
                list(
                    utils.paginate_endpoint(
                        self._datasets_api.get_datasets_query_by_name,
                        dataset_name=dataset_name,
                        exact=True,
                        shared=False,
                    )
                )
            )
        if shared or shared is None:
            datasets.extend(
                list(
                    utils.paginate_endpoint(
                        self._datasets_api.get_datasets_query_by_name,
                        dataset_name=dataset_name,
                        exact=True,
                        shared=True,
                    )
                )
            )
            datasets.extend(
                list(
                    utils.paginate_endpoint(
                        self._datasets_api.get_datasets_query_by_name,
                        dataset_name=dataset_name,
                        exact=True,
                        get_assets_of_team=True,
                    )
                )
            )

        # De-duplicate datasets because results from shared=True and
        # those from get_assets_of_team=True might overlap
        dataset_ids: Set[str] = set()
        filtered_datasets: List[DatasetData] = []
        for dataset in datasets:
            if dataset.id not in dataset_ids:
                dataset_ids.add(dataset.id)
                filtered_datasets.append(dataset)

        return filtered_datasets

    def get_datasets_iter(
        self, shared: Optional[bool] = False
    ) -> Iterator[DatasetData]:
        """Returns an iterator over all datasets owned by the current user.

        There can be multiple datasets with the same name accessible to the current
        user. This can happen if either:
        * A dataset has been explicitly shared with the user
        * The user has access to team datasets
        The `shared` flag controls whether these datasets are returned.

        Args:
            shared:
                * If False (default), returns only datasets owned by the user. In this
                case at most one dataset will be returned.
                * If True, returns datasets which have been shared with the user,
                including team datasets. Excludes user's own datasets. Can return
                multiple datasets.
                * If None, returns all datasets the users has access to. Can return
                multiple datasets.

        Returns:
            An iterator over datasets owned by the current user.
        """
        dataset_iterable: Iterator[DatasetData] = (_ for _ in ())
        if not shared or shared is None:
            dataset_iterable = utils.paginate_endpoint(
                self._datasets_api.get_datasets,
                shared=False,
            )
        if shared or shared is None:
            dataset_iterable = chain(
                dataset_iterable,
                utils.paginate_endpoint(
                    self._datasets_api.get_datasets,
                    shared=True,
                ),
            )
            dataset_iterable = chain(
                dataset_iterable,
                utils.paginate_endpoint(
                    self._datasets_api.get_datasets,
                    get_assets_of_team=True,
                ),
            )

        # De-duplicate datasets because results from shared=True and
        # those from get_assets_of_team=True might overlap
        dataset_ids: Set[str] = set()
        for dataset in dataset_iterable:
            if dataset.id not in dataset_ids:
                dataset_ids.add(dataset.id)
                yield dataset

    def get_datasets(self, shared: Optional[bool] = False) -> List[DatasetData]:
        """Returns all datasets owned by the current user.

        There can be multiple datasets with the same name accessible to the current
        user. This can happen if either:
        * A dataset has been explicitly shared with the user
        * The user has access to team datasets
        The `shared` flag controls whether these datasets are returned.

        Args:
            shared:
                * If False (default), returns only datasets owned by the user. In this
                case at most one dataset will be returned.
                * If True, returns datasets which have been shared with the user,
                including team datasets. Excludes user's own datasets. Can return
                multiple datasets.
                * If None, returns all datasets the users has access to. Can return
                multiple datasets.

        Returns:
            A list of datasets owned by the current user.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> client.create_dataset("your-dataset-name", dataset_type=DatasetType.IMAGES)
            >>> client.get_datasets()
            [{'created_at': 1685009504596,
             'datasource_processed_until_timestamp': 1685009513,
             'datasources': ['646f346004d77b4e1424e67e', '646f346004d77b4e1424e695'],
             'id': '646f34608a5613b57d8b73c9',
             'img_type': 'full',
             'type': 'Images',
             ...}]
        """
        return list(self.get_datasets_iter(shared))

    def get_all_datasets(self) -> List[DatasetData]:
        """Returns all datasets the user has access to.

        DEPRECATED in favour of get_datasets(shared=None) and will be removed in the
        future.
        """
        warnings.warn(
            "get_all_datasets() is deprecated in favour of get_datasets(shared=None) "
            "and will be removed in the future.",
            DeprecationWarning,
        )
        owned_datasets = self.get_datasets(shared=None)
        return owned_datasets

    def set_dataset_id_by_name(
        self, dataset_name: str, shared: Optional[bool] = False
    ) -> None:
        """Sets the dataset ID in the API client given the name of the desired dataset.

        There can be multiple datasets with the same name accessible to the current
        user. This can happen if either:
        * A dataset has been explicitly shared with the user
        * The user has access to team datasets
        The `shared` flag controls whether these datasets are also checked. If multiple
        datasets with the given name are found, the API client uses the ID of the first
        dataset and prints a warning message.

        Args:
            dataset_name:
                The name of the target dataset.
            shared:
                * If False (default), checks only datasets owned by the user.
                * If True, returns datasets which have been shared with the user,
                including team datasets. Excludes user's own datasets. There can be
                multiple candidate datasets.
                * If None, returns all datasets the users has access to. There can be
                multiple candidate datasets.

        Raises:
            ValueError:
                If no dataset with the given name exists.

        Examples:
            >>> # A new session. Dataset "old-dataset" was created before.
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> client.set_dataset_id_by_name("old-dataset")
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
    ) -> None:
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

        Raises:
            ValueError: If a dataset with dataset_name already exists.

        Examples:
            >>> from lightly.api import ApiWorkflowClient
            >>> from lightly.openapi_generated.swagger_client.models import DatasetType
            >>>
            >>> client = ApiWorkflowClient(token="YOUR_TOKEN")
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
        )

    def _create_dataset_without_check_existing(
        self, dataset_name: str, dataset_type: str
    ) -> None:
        """Creates a dataset on the Lightly Platform.

        No checking if a dataset with such a name already exists is performed.

        Args:
            dataset_name:
                The name of the dataset to be created.
            dataset_type:
                The type of the dataset. We recommend to use the API provided
                constants `DatasetType.IMAGES` and `DatasetType.VIDEOS`.

        """
        body = DatasetCreateRequest(
            name=dataset_name, type=dataset_type, creator=self._creator
        )
        response: CreateEntityResponse = self._datasets_api.create_dataset(
            dataset_create_request=body
        )
        self._dataset_id = response.id

    def create_new_dataset_with_unique_name(
        self,
        dataset_basename: str,
        dataset_type: str = DatasetType.IMAGES,
    ) -> None:
        """Creates a new dataset on the Lightly Platform.

        If a dataset with the specified name already exists,
        the name is suffixed by a counter value.

        Args:
            dataset_basename:
                The name of the dataset to be created.
            dataset_type:
                The type of the dataset. We recommend to use the API provided
                constants `DatasetType.IMAGES` and `DatasetType.VIDEOS`.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Create a dataset with a brand new name.
            >>> client.create_new_dataset_with_unique_name("new-dataset")
            >>> client.get_dataset_by_id(client.dataset_id)
            {'id': '6470abef4f0eb7e635c30954',
             'name': 'new-dataset',
             ...}
            >>>
            >>> # Create another dataset with the same name. This time, the
            >>> # new dataset should have a suffix `_1`.
            >>> client.create_new_dataset_with_unique_name("new-dataset")
            >>> client.get_dataset_by_id(client.dataset_id)
            {'id': '6470ac194f0eb7e635c30990',
             'name': 'new-dataset_1',
             ...}

        """
        if not self.dataset_name_exists(dataset_name=dataset_basename):
            self._create_dataset_without_check_existing(
                dataset_name=dataset_basename,
                dataset_type=dataset_type,
            )
        else:
            existing_datasets = list(
                utils.paginate_endpoint(
                    self._datasets_api.get_datasets_query_by_name,
                    dataset_name=dataset_basename,
                    exact=False,
                    shared=False,
                )
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
            )

    def delete_dataset_by_id(self, dataset_id: str) -> None:
        """Deletes a dataset on the Lightly Platform.

        Args:
            dataset_id:
                The ID of the dataset to be deleted.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> client.create_dataset("your-dataset-name", dataset_type=DatasetType.IMAGES)
            >>> dataset_id = client.dataset_id
            >>> client.dataset_exists(dataset_id=dataset_id)
            True
            >>>
            >>> # Delete the dataset
            >>> client.delete_dataset_by_id(dataset_id=dataset_id)
            >>> client.dataset_exists(dataset_id=dataset_id)
            False
        """
        self._datasets_api.delete_dataset_by_id(dataset_id=dataset_id)
        del self._dataset_id
