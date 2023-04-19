import copy
import dataclasses
import difflib
import time
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Type, TypeVar, Union

from lightly.api import utils
from lightly.api.utils import retry
from lightly.openapi_generated.swagger_client import (
    ApiClient,
    CreateDockerWorkerRegistryEntryRequest,
    DockerRunData,
    DockerRunScheduledCreateRequest,
    DockerRunScheduledData,
    DockerRunScheduledPriority,
    DockerRunScheduledState,
    DockerRunState,
    DockerWorkerConfigV3,
    DockerWorkerConfigV3CreateRequest,
    DockerWorkerConfigV3Docker,
    DockerWorkerConfigV3Lightly,
    DockerWorkerRegistryEntryData,
    DockerWorkerType,
    SelectionConfig,
    SelectionConfigEntry,
    SelectionConfigEntryInput,
    SelectionConfigEntryStrategy,
    TagData,
)
from lightly.openapi_generated.swagger_client.rest import ApiException

STATE_SCHEDULED_ID_NOT_FOUND = "CANCELED_OR_NOT_EXISTING"


class InvalidConfigurationError(RuntimeError):
    pass


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
            DockerRunState.CRASHED,
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
    def register_compute_worker(
        self, name: str = "Default", labels: Optional[List[str]] = None
    ) -> str:
        """Registers a new compute worker.

        If a worker with the same name already exists, the worker id of the existing
        worker is returned instead of registering a new worker.

        Args:
            name:
                The name of the Lightly Worker.
            labels:
                The labels of the Lightly Worker.
                See our docs for more information regarding the labels parameter:
                https://docs.lightly.ai/docs/assign-scheduled-runs-to-specific-workers

        Returns:
            The id of the newly registered compute worker.

        """
        if labels is None:
            labels = []
        request = CreateDockerWorkerRegistryEntryRequest(
            name=name,
            worker_type=DockerWorkerType.FULL,
            labels=labels,
            creator=self._creator,
        )
        response = self._compute_worker_api.register_docker_worker(request)
        return response.id

    def get_compute_worker_ids(self) -> List[str]:
        """Fetches the IDs of all registered compute workers.

        Returns:
            A list of worker IDs.
        """
        entries = self._compute_worker_api.get_docker_worker_registry_entries()
        return [entry.id for entry in entries]

    def get_compute_workers(self) -> List[DockerWorkerRegistryEntryData]:
        """Fetches details of all registered compute workers.

        Returns:
            A list of compute workers.
        """
        entries: list[
            DockerWorkerRegistryEntryData
        ] = self._compute_worker_api.get_docker_worker_registry_entries()
        return entries

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

        See our docs for more information regarding the different configurations:
        https://docs.lightly.ai/docs/all-configuration-options

        Args:
            worker_config:
                Compute worker configuration.
            lightly_config:
                Lightly configuration.
            selection_config:
                Selection configuration.

        Returns:
            The id of the created config.

        """
        if isinstance(selection_config, dict):
            selection = selection_config_from_dict(cfg=selection_config)
        else:
            selection = selection_config

        if worker_config is not None:
            worker_config_cc = _config_to_camel_case(cfg=worker_config)
            deserialize_worker_config = _get_deserialize(
                api_client=self.api_client,
                klass=DockerWorkerConfigV3Docker,
            )
            docker = deserialize_worker_config(worker_config_cc)
            _validate_config(cfg=worker_config, obj=docker)
        else:
            docker = None

        if lightly_config is not None:
            lightly_config_cc = _config_to_camel_case(cfg=lightly_config)
            deserialize_lightly_config = _get_deserialize(
                api_client=self.api_client,
                klass=DockerWorkerConfigV3Lightly,
            )
            lightly = deserialize_lightly_config(lightly_config_cc)
            _validate_config(cfg=lightly_config, obj=lightly)
        else:
            lightly = None

        config = DockerWorkerConfigV3(
            worker_type=DockerWorkerType.FULL,
            docker=docker,
            lightly=lightly,
            selection=selection,
        )
        request = DockerWorkerConfigV3CreateRequest(
            config=config, creator=self._creator
        )
        response = self._compute_worker_api.create_docker_worker_config_v3(request)
        return response.id

    def schedule_compute_worker_run(
        self,
        worker_config: Optional[Dict[str, Any]] = None,
        lightly_config: Optional[Dict[str, Any]] = None,
        selection_config: Optional[Union[Dict[str, Any], SelectionConfig]] = None,
        priority: str = DockerRunScheduledPriority.MID,
        runs_on: Optional[List[str]] = None,
    ) -> str:
        """Schedules a run with the given configurations.

        See our docs for more information regarding the different configurations:
        https://docs.lightly.ai/docs/all-configuration-options

        Args:
            worker_config:
                Compute worker configuration.
            lightly_config:
                Lightly configuration.
            selection_config:
                Selection configuration.
            runs_on:
                The required labels the Lightly Worker must have to take the job.
                See our docs for more information regarding the runs_on paramter:
                https://docs.lightly.ai/docs/assign-scheduled-runs-to-specific-workers

        Returns:
            The id of the scheduled run.

        Raises:
            ApiException:
                If the API call returns a status code other than 200.
                    400: Missing or invalid parameters
                    402: Insufficient plan
                    403: Not authorized for this resource or invalid token
                    404: Resource (dataset or config) not found
                    422: Missing or invalid file in datasource
            InvalidConfigError:
                If one of the configurations is invalid.

        """
        if runs_on is None:
            runs_on = []
        config_id = self.create_compute_worker_config(
            worker_config=worker_config,
            lightly_config=lightly_config,
            selection_config=selection_config,
        )
        request = DockerRunScheduledCreateRequest(
            config_id=config_id,
            priority=priority,
            runs_on=runs_on,
            creator=self._creator,
        )
        response = self._compute_worker_api.create_docker_run_scheduled_by_dataset_id(
            body=request,
            dataset_id=self.dataset_id,
        )
        return response.id

    def get_compute_worker_runs(
        self,
        dataset_id: Optional[str] = None,
    ) -> List[DockerRunData]:
        """Get all compute worker runs for the user.

        Args:
            dataset_id:
                If set, then only runs for the given dataset are returned.

        Returns:
            Runs sorted by creation time from old to new.

        """
        if dataset_id is not None:
            runs: List[DockerRunData] = utils.paginate_endpoint(
                self._compute_worker_api.get_docker_runs_query_by_dataset_id,
                dataset_id=dataset_id,
            )
        else:
            runs: List[DockerRunData] = utils.paginate_endpoint(
                self._compute_worker_api.get_docker_runs,
            )
        sorted_runs = sorted(runs, key=lambda run: run.created_at or -1)
        return sorted_runs

    def get_compute_worker_run(self, run_id: str) -> DockerRunData:
        """Returns a run given its id.

        Raises:
            ApiException:
                If no run with the given id exists.
        """
        return self._compute_worker_api.get_docker_run_by_id(run_id=run_id)

    def get_compute_worker_run_from_scheduled_run(
        self,
        scheduled_run_id: str,
    ) -> DockerRunData:
        """Returns a run given its scheduled run id.

        Raises:
            ApiException:
                If no run with the given scheduled run id exists or if the scheduled
                run has not yet started being processed by a worker.
        """
        return self._compute_worker_api.get_docker_run_by_scheduled_id(
            scheduled_id=scheduled_run_id
        )

    def get_scheduled_compute_worker_runs(
        self,
        state: Optional[str] = None,
    ) -> List[DockerRunScheduledData]:
        """Returns a list of all scheduled compute worker runs for the current
        dataset.

        Args:
            state:
                DockerRunScheduledState value. If specified, then only runs in the given
                state are returned. If omitted, then runs which have not yet finished
                (neither 'DONE' nor 'CANCELED') are returned. Valid states are 'OPEN',
                'LOCKED', 'DONE', and 'CANCELED'.
        """
        if state is not None:
            return self._compute_worker_api.get_docker_runs_scheduled_by_dataset_id(
                dataset_id=self.dataset_id,
                state=state,
            )
        return self._compute_worker_api.get_docker_runs_scheduled_by_dataset_id(
            dataset_id=self.dataset_id,
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

    def get_compute_worker_run_tags(self, run_id: str) -> List[TagData]:
        """Returns all tags from a run for the current dataset.

        Only returns tags for runs made with Lightly Worker version >=2.4.2.

        Args:
            run_id:
                Run id from which to return tags.

        Returns:
            List of tags created by the run. The tags are ordered by creation date from
            newest to oldest.

        Examples:
            >>> # Get filenames from last run.
            >>>
            >>> from lightly.api import ApiWorkflowClient
            >>> client = ApiWorkflowClient(
            >>>     token="MY_LIGHTLY_TOKEN", dataset_id="MY_DATASET_ID"
            >>> )
            >>> tags = client.get_compute_worker_run_tags(run_id="MY_LAST_RUN_ID")
            >>> filenames = client.export_filenames_by_tag_name(tag_name=tags[0].name)

        """
        tags = self._compute_worker_api.get_docker_run_tags(run_id=run_id)
        tags_in_dataset = [tag for tag in tags if tag.dataset_id == self.dataset_id]
        return tags_in_dataset


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


_T = TypeVar("_T")


def _get_deserialize(
    api_client: ApiClient,
    klass: Type[_T],
) -> Callable[[Dict[str, Any]], _T]:
    """Returns the deserializer of the ApiClient class for class klass.

    TODO(Philipp, 02/23): We should replace this by our own deserializer which
    accepts snake case strings as input.

    The deserializer takes a dictionary and and returns an instance of klass.

    """
    deserialize = getattr(api_client, "_ApiClient__deserialize")
    return partial(deserialize, klass=klass)


def _config_to_camel_case(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Converts all keys in the cfg dictionary to camelCase."""
    cfg_camel_case = {}
    for key, value in cfg.items():
        key_camel_case = _snake_to_camel_case(key)
        if isinstance(value, dict):
            cfg_camel_case[key_camel_case] = _config_to_camel_case(value)
        else:
            cfg_camel_case[key_camel_case] = value
    return cfg_camel_case


def _snake_to_camel_case(snake: str) -> str:
    """Converts the snake_case input to camelCase."""
    components = snake.split("_")
    return components[0] + "".join(component.title() for component in components[1:])


def _validate_config(
    cfg: Optional[Dict[str, Any]],
    obj: Any,
) -> None:
    """Validates that all keys in cfg are legitimate configuration options.

    Recursively checks if the keys in the cfg dictionary match the attributes of
    the DockerWorkerConfigV2Docker/DockerWorkerConfigV2Lightly instances. If not,
    suggests a best match based on the keys in 'swagger_types'.

    Raises:
        TypeError: If obj is not of swagger type.

    """

    if cfg is None:
        return

    if not hasattr(type(obj), "swagger_types"):
        raise TypeError(
            f"Type {type(obj)} of argument 'obj' has not attribute 'swagger_types'"
        )

    for key, item in cfg.items():
        if not hasattr(obj, key):
            possible_options = list(type(obj).swagger_types.keys())
            closest_match = difflib.get_close_matches(
                word=key, possibilities=possible_options, n=1, cutoff=0.0
            )[0]
            error_msg = (
                f"Option '{key}' does not exist! Did you mean '{closest_match}'?"
            )
            raise InvalidConfigurationError(error_msg)
        if isinstance(item, dict):
            _validate_config(item, getattr(obj, key))
