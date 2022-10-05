import copy
import dataclasses
import time
from typing import Any, Dict, List, Optional, Union, Iterator

from lightly.api.utils import retry
from lightly.openapi_generated.swagger_client import (
    DockerRunScheduledState,
    DockerRunState,
)
from lightly.openapi_generated.swagger_client import (
    SelectionConfig,
    SelectionConfigEntryInput,
    SelectionConfigEntryStrategy,
    SelectionConfigEntry,
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
from lightly.openapi_generated.swagger_client.models.docker_worker_config import (
    DockerWorkerConfig,
)
from lightly.openapi_generated.swagger_client.models.docker_worker_config_create_request import (
    DockerWorkerConfigCreateRequest,
)
from lightly.openapi_generated.swagger_client.models.docker_worker_type import (
    DockerWorkerType,
)
from lightly.openapi_generated.swagger_client.rest import ApiException

STATE_SCHEDULED_ID_NOT_FOUND = "CANCELED_OR_NOT_EXISTING"


@dataclasses.dataclass
class ComputeWorkerRunInfo:
    """
    Contains information about a compute worker run that is useful for monitoring it.

    Attributes:
        state:
            The state of the compute worker run.
        message:
            The last message of the compute worker run.
    """

    state: Union[
        DockerRunState, DockerRunScheduledState.OPEN, STATE_SCHEDULED_ID_NOT_FOUND
    ]
    message: str

    def in_end_state(self) -> bool:
        """Returns wether the compute worker has ended"""
        return self.state in [
            DockerRunState.COMPLETED,
            DockerRunState.ABORTED,
            DockerRunState.FAILED,
            STATE_SCHEDULED_ID_NOT_FOUND,
        ]

    def ended_successfully(self) -> bool:
        """
        Returns wether the compute worker ended successfully or failed.
        Raises a ValueError if the compute worker is still running.
        """
        if not self.in_end_state():
            raise ValueError("Compute worker is still running")
        return self.state == DockerRunState.COMPLETED


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

    def _get_scheduled_run_by_id(self, scheduled_run_id: str) -> DockerRunScheduledData:
        """Returns the schedule run data given the id of the scheduled run.

        TODO (MALTE, 09/2022): Have a proper API endpoint for doing this.
        Args:
            scheduled_run_id:
                The id with which the run was scheduled.

        Returns:
            Data about the scheduled run.

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

        Returns:
            Data about the compute worker run.

        Examples:
            >>> # Scheduled a compute worker run and get its state
            >>> scheduled_run_id = client.schedule_compute_worker_run(...)
            >>> run_info = client.get_compute_worker_run_info(scheduled_run_id)
            >>> print(run_info)

        """
        """
        Because we currently (09/2022) have different Database entries for a ScheduledRun and DockerRun,
        the logic is more complicated and covers three cases.
        """
        try:
            # Case 1: DockerRun exists.
            docker_run: DockerRunData = (
                self._compute_worker_api.get_docker_run_by_scheduled_id(
                    scheduled_id=scheduled_run_id
                )
            )
            info = ComputeWorkerRunInfo(
                state=docker_run.state, message=docker_run.message
            )
        except ApiException:
            try:
                # Case 2: DockerRun does NOT exist, but ScheduledRun exists.
                _ = self._get_scheduled_run_by_id(scheduled_run_id)
                info = ComputeWorkerRunInfo(
                    state=DockerRunScheduledState.OPEN,
                    message="Waiting for pickup by Lightly Worker. "
                    "Make sure to start a Lightly Worker connected to your "
                    "user token to process the job.",
                )
            except ApiException:
                # Case 3: NEITHER the DockerRun NOR the ScheduledRun exist.
                info = ComputeWorkerRunInfo(
                    state=STATE_SCHEDULED_ID_NOT_FOUND,
                    message=f"Could not find a job for the given run_id: '{scheduled_run_id}'. "
                    "The scheduled run does not exist or was canceled before "
                    "being picked up by a Lightly Worker.",
                )
        return info

    def compute_worker_run_info_generator(
        self, scheduled_run_id: str
    ) -> Iterator[ComputeWorkerRunInfo]:
        """
        Yields information about a compute worker run

        Polls the compute worker status every 30s.
        If the status changed, it will yield a new ComputeWorkerRunInfo.
        If the compute worker run finished, the generator stops.

        Args:
            scheduled_run_id:
                The id with which the run was scheduled.

        Returns:
            Generator of information about the compute worker run status.

        Examples:
            >>> # Scheduled a compute worker run and monitor its state
            >>> scheduled_run_id = client.schedule_compute_worker_run(...)
            >>> for run_info in client.compute_worker_run_info_generator(scheduled_run_id):
            >>>     print(f"Compute worker run is now in state='{run_info.state}' with message='{run_info.message}'")
            >>>

        """
        last_run_info = None
        while True:
            run_info = self.get_compute_worker_run_info(
                scheduled_run_id=scheduled_run_id
            )

            # Only yield new run_info
            if run_info != last_run_info:
                yield run_info

            # Break if the scheduled run is in one of the end states.
            if run_info.in_end_state():
                break

            # Wait before polling the state again
            time.sleep(30)  # Keep this at 30s or larger to prevent rate limiting.

            last_run_info = run_info


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
