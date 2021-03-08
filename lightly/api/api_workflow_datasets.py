from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightly.api.api_workflow_client import ApiWorkflowClient

from typing import List

from lightly.openapi_generated.swagger_client.models.create_entity_response import CreateEntityResponse
from lightly.openapi_generated.swagger_client.models.dataset_create_request import DatasetCreateRequest
from lightly.openapi_generated.swagger_client.models.dataset_data import DatasetData


class _DatasetsMixin:

    def set_dataset_id_by_name(self: ApiWorkflowClient, dataset_name: str):
        """Sets the dataset id given the name of the dataset

        Args:
            dataset_name:
                The name of the dataset for which the dataset_id
                should be set as attribute

        Raises: ValueError

        """
        current_datasets: List[DatasetData] \
            = self.datasets_api.get_datasets()

        try:
            dataset_with_specified_name = next(dataset for dataset in current_datasets if dataset.name == dataset_name)
            self._dataset_id = dataset_with_specified_name.id
        except StopIteration:
            raise ValueError(f"A dataset with the name {dataset_name} does not exist on the web platform. "
                             f"Please create it first.")

    def create_dataset(self: ApiWorkflowClient, dataset_name: str):
        """Creates a dataset on the webplatform

        If a dataset with that name already exists, instead the dataset_id is set.
        Args:
            dataset_name:
                The name of the dataset to be created.

        """
        try:
            self.set_dataset_id_by_name(dataset_name)
        except ValueError:
            self._create_dataset_without_check_existing(dataset_name=dataset_name)

    def _create_dataset_without_check_existing(self: ApiWorkflowClient, dataset_name: str):
        """Creates a dataset on the webplatform

        No checking if a dataset with such a name already exists is performed.
        Args:
            dataset_name:
                The name of the dataset to be created.

        """
        body = DatasetCreateRequest(name=dataset_name)
        response: CreateEntityResponse = self.datasets_api.create_dataset(body=body)
        self._dataset_id = response.id

    def create_new_dataset_with_counter(self: ApiWorkflowClient, dataset_basename: str):
        """Creates a new dataset on the web platform

        If a dataset with the specified name already exists,
        a counter is added to the name to be able to still create it.
        Args:
            dataset_basename:
                The name of the dataset to be created.

        """
        current_datasets: List[DatasetData] \
            = self.datasets_api.get_datasets()
        current_datasets_names = [dataset.name for dataset in current_datasets]

        if dataset_basename not in current_datasets_names:
            self._create_dataset_without_check_existing(dataset_name=dataset_basename)
            return

        counter = 1
        dataset_name = f"{dataset_basename}_{counter}"
        while dataset_name in current_datasets_names:
            counter += 1
            dataset_name = f"{dataset_basename}_{counter}"
        self._create_dataset_without_check_existing(dataset_name=dataset_name)

    def delete_dataset_by_id(self: ApiWorkflowClient, dataset_id: str):
        """Deletes a dataset on the web platform

        Args:
            dataset_id:
                The id of the dataset to be deleted.

        """
        self.datasets_api.delete_dataset_by_id(dataset_id=dataset_id)
        del self._dataset_id
