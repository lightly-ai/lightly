import copy
import dataclasses
from typing import Any, Dict, List, Optional, Union, Tuple, Literal

from lightly.api.utils import retry
from lightly.openapi_generated.swagger_client import (
    SelectionConfig,
    DockerRunScheduledState,
    DockerRunState,
)
from lightly.openapi_generated.swagger_client.models.create_docker_worker_registry_entry_request import (
    CreateDockerWorkerRegistryEntryRequest,
)
from lightly.openapi_generated.swagger_client.models.docker_run_data import (
    DockerRunData,
)
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_create_request import (
    DockerRunScheduledCreateRequest,
)
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_data import (
    DockerRunScheduledData,
)
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_priority import (
    DockerRunScheduledPriority,
)
from lightly.openapi_generated.swagger_client.models.docker_worker_type import (
    DockerWorkerType,
)
from lightly.openapi_generated.swagger_client.models.docker_worker_config import (
    DockerWorkerConfig,
)
from lightly.openapi_generated.swagger_client.models.docker_worker_config_create_request import (
    DockerWorkerConfigCreateRequest,
)
from lightly.openapi_generated.swagger_client import (
    SelectionConfig,
    SelectionConfigEntryInput,
    SelectionConfigEntryStrategy,
    SelectionConfigEntry,
)
from lightly.openapi_generated.swagger_client.rest import ApiException


STATE_SCHEDULED_ID_NOT_FOUND = "CANCELED_OR_NOT_EXISTING"

@dataclasses.dataclass
class ComputeWorkerRunInfo:
    state: Union[DockerRunState, DockerRunScheduledState.OPEN, STATE_SCHEDULED_ID_NOT_FOUND]
    message: str

    def in_end_state(self):
        return self.state in [DockerRunState.ABORTED, DockerRunState.COMPLETED, DockerRunState.FAILED, STATE_SCHEDULED_ID_NOT_FOUND]


class _ComputeWorkerMixin:
    def register_compute_worker(self, name: str = "Default") -> str:
        """Registers a new compute worker.

        Args:
            name:
                The name of the compute worker.

        Returns:
            The id of the newly registered compute worker.

        """
        request = CreateDockerWorkerRegistryEntryRequest(
            name=name, worker_type=DockerWorkerType.FULL
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
        selection_config: Optional[Union[Dict[str, Any], SelectionConfig]] = None,
    ) -> str:
        """Creates a new configuration for a compute worker run.

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
            The id of the created config.

        """
        if isinstance(selection_config, dict):
            selection = selection_config_from_dict(cfg=selection_config)
        else:
            selection = selection_config
        config = DockerWorkerConfig(
            worker_type=DockerWorkerType.FULL,
            docker=worker_config,
            lightly=lightly_config,
            selection=selection,
        )
        request = DockerWorkerConfigCreateRequest(config)
        response = self._compute_worker_api.create_docker_worker_config(request)
        return response.id

    def schedule_compute_worker_run(
        self,
        worker_config: Optional[Dict[str, Any]] = None,
        lightly_config: Optional[Dict[str, Any]] = None,
        selection_config: Optional[Union[Dict[str, Any], SelectionConfig]] = None,
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
                https://docs.lightly.ai/docker/getting_started/selection.html

        Returns:
            The id of the scheduled run.

        """
        config_id = self.create_compute_worker_config(
            worker_config=worker_config,
            lightly_config=lightly_config,
            selection_config=selection_config,
        )
        request = DockerRunScheduledCreateRequest(
            config_id=config_id, priority=priority
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

    def get_scheduled_run_by_id(self, scheduled_run_id: str) -> DockerRunScheduledData:
        """Returns the schedule run data given the id of the scheduled run.

        TODO (MALTE, 09/2022): Have a proper API endpoint for doing this.
        """
        try:
            run: DockerRunScheduledData = next(
                run
                for run in retry(
                    lambda: self._compute_worker_api.get_docker_runs_scheduled_by_dataset_id(
                        self.dataset_id
                    )
                )
                if run.id == scheduled_run_id
            )
            return run
        except StopIteration:
            raise ApiException(
                f"No scheduled run found for run with scheduled_run_id='{scheduled_run_id}'."
            )

    def get_compute_worker_run_info(
        self, scheduled_run_id: str
    ) -> ComputeWorkerRunInfo:
        """Returns information about the compute worker run.

        Args:
            scheduled_run_id:
                The id with which the run was scheduled.

        """
        try:
            docker_run: DockerRunData = self._compute_worker_api.get_docker_run_by_scheduled_id(scheduled_id=scheduled_run_id)
            info = ComputeWorkerRunInfo(state=docker_run.state, message=docker_run.message)
        except ApiException:
            try:
                _ = self.get_scheduled_run_by_id(scheduled_run_id)
                info = ComputeWorkerRunInfo(state=DockerRunScheduledState.OPEN, message="Waiting for pickup by compute worker.")
            except ApiException:
                info = ComputeWorkerRunInfo(state=STATE_SCHEDULED_ID_NOT_FOUND, message="The scheduled run was either canceled or does not exist.")
        return info


def selection_config_from_dict(cfg: Dict[str, Any]) -> SelectionConfig:
    """Recursively converts selection config from dict to a SelectionConfig instance."""
    new_cfg = copy.deepcopy(cfg)
    strategies = []
    for entry in new_cfg.get("strategies", []):
        entry["input"] = SelectionConfigEntryInput(**entry["input"])
        entry["strategy"] = SelectionConfigEntryStrategy(**entry["strategy"])
        strategies.append(SelectionConfigEntry(**entry))
    new_cfg["strategies"] = strategies
    return SelectionConfig(**new_cfg)
