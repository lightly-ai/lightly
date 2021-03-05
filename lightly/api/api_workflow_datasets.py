from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightly.api.api_workflow_client import ApiWorkflowClient

from typing import List

from lightly.openapi_generated.swagger_client.models.create_entity_response import CreateEntityResponse
from lightly.openapi_generated.swagger_client.models.dataset_create_request import DatasetCreateRequest
from lightly.openapi_generated.swagger_client.models.dataset_data import DatasetData


class _DatasetsMixin:

    def set_dataset_id(self: ApiWorkflowClient, dataset_name: str):
        current_datasets: List[DatasetData] \
            = self.datasets_api.get_datasets()

        try:
            dataset_with_specified_name = next(dataset for dataset in current_datasets if dataset.name == dataset_name)
            self._dataset_id = dataset_with_specified_name.id
        except StopIteration:
            raise ValueError(f"A dataset with the name {dataset_name} does not exist on the web platform. "
                             f"Please create it first.")

    def create_dataset(self: ApiWorkflowClient, dataset_name: str):
        try:
            self.set_dataset_id(dataset_name)
        except ValueError:
            self._create_dataset_without_check_existing(dataset_name=dataset_name)

    def _create_dataset_without_check_existing(self: ApiWorkflowClient, dataset_name: str):
        body = DatasetCreateRequest(name=dataset_name)
        response: CreateEntityResponse = self.datasets_api.create_dataset(body=body)
        self._dataset_id = response.id

    def create_new_dataset_with_counter(self, dataset_basename: str):
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

    def delete_dataset(self: ApiWorkflowClient):
        self.datasets_api.delete_dataset_by_id(dataset_id=self.dataset_id)
