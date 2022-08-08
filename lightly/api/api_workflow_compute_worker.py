from typing import Any, Dict, List, Optional

from lightly.openapi_generated.swagger_client.models.create_docker_worker_registry_entry_request import CreateDockerWorkerRegistryEntryRequest
from lightly.openapi_generated.swagger_client.models.docker_run_data import DockerRunData
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_create_request import DockerRunScheduledCreateRequest
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_data import DockerRunScheduledData
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_priority import DockerRunScheduledPriority
from lightly.openapi_generated.swagger_client.models.docker_worker_type import DockerWorkerType
from lightly.openapi_generated.swagger_client.models.docker_worker_config import DockerWorkerConfig
from lightly.openapi_generated.swagger_client.models.docker_worker_config_create_request import DockerWorkerConfigCreateRequest


class _ComputeWorkerMixin:

    def register_compute_worker(self, name: str = 'Default') -> str:
        """Registers a new compute worker.
        
        Args:
            name:
                The name of the compute worker.
        
        Returns:
            The id of the newly registered compute worker.

        """
        request = CreateDockerWorkerRegistryEntryRequest(
            name=name, 
            worker_type=DockerWorkerType.FULL
        )
        response = self._compute_worker_api.register_docker_worker(request)
        return response.id

    def get_compute_worker_ids(self) -> List[str]:
        """Returns the ids of all registered compute workers."""
        entries = self._compute_worker_api.get_docker_worker_registry_entries()
        return [entry.id for entry in entries]

    def delete_compute_worker(self, worker_id: str):
        """Removes a compute worker.
        
        Args:
            worker_id:
                The id of the worker to remove.

        """
        self._compute_worker_api.delete_docker_worker_registry_entry_by_id(worker_id)

    def create_compute_worker_config(
        self, 
        worker_config: Optional[Dict[str, Any]] = None, 
        lightly_config: Optional[Dict[str, Any]] = None,
        selection_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Creates a new configuration for a compute worker run.
        
        Args:
            worker_config:
                Compute worker configuration. All possible values are listed in 
                our docs: https://docs.lightly.ai/docker/configuration/configuration.html#list-of-parameters
            lightly_config:
                Lightly configuration. All possible values are listed in our
                docs: https://docs.lightly.ai/lightly.cli.html#default-settings

        Returns:
            The id of the created config.

        """
        config = DockerWorkerConfig(
            worker_type=DockerWorkerType.FULL, 
            docker=worker_config, 
            lightly=lightly_config,
            selection=selection_config,
        )
        request = DockerWorkerConfigCreateRequest(config)
        response = self._compute_worker_api.create_docker_worker_config(request)
        return response.id

    def schedule_compute_worker_run(
        self,
        worker_config: Optional[Dict[str, Any]] = None,
        lightly_config: Optional[Dict[str, Any]] = None,
        selection_config: Optional[Dict[str, Any]] = None,
        priority: str = DockerRunScheduledPriority.MID,
    ) -> str:
        """Schedules a run with the given configurations.
        
        Args:
            worker_config:
                Compute worker configuration. All possible values are listed in 
                our docs: https://docs.lightly.ai/docker/configuration/configuration.html#list-of-parameters
            lightly_config:
                Lightly configuration. All possible values are listed in our
                docs: https://docs.lightly.ai/lightly.cli.html#default-settings
            selection_config:
                Selection configuration. See the docs for more information:
                TODO: add link

        Returns:
            The id of the scheduled run.

        """
        config_id = self.create_compute_worker_config(
            worker_config=worker_config,
            lightly_config=lightly_config,
            selection_config=selection_config,
        )
        request = DockerRunScheduledCreateRequest(
            config_id=config_id, 
            priority=priority
        )
        response = self._compute_worker_api.create_docker_run_scheduled_by_dataset_id(
            body=request,
            dataset_id=self.dataset_id,
        )
        return response.id

    def get_compute_worker_runs(self) -> List[DockerRunData]:
        """Returns all compute worker runs for the user."""
        return self._compute_worker_api.get_docker_runs()

    def get_scheduled_compute_worker_runs(
        self,
    ) -> List[DockerRunScheduledData]:
        """Returns a list of all scheduled compute worker runs for the current
        dataset.
        """
        return self._compute_worker_api.get_docker_runs_scheduled_by_dataset_id(
            dataset_id=self.dataset_id
        )
