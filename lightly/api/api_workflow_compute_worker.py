import copy
import dataclasses
import difflib
import json
import time
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Type, TypeVar, Union

from lightly.api import utils
from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.models import (
    AutoTask,
    AutoTaskTiling,
    CreateDockerWorkerRegistryEntryRequest,
    DockerRunData,
    DockerRunScheduledCreateRequest,
    DockerRunScheduledData,
    DockerRunScheduledPriority,
    DockerRunScheduledState,
    DockerRunState,
    DockerWorkerConfigOmniVXCreateRequest,
    DockerWorkerConfigV3Lightly,
    DockerWorkerConfigV4,
    DockerWorkerConfigV4Docker,
    DockerWorkerRegistryEntryData,
    DockerWorkerType,
    SelectionConfigV4,
    SelectionConfigV4Entry,
    SelectionConfigV4EntryInput,
    SelectionConfigV4EntryStrategy,
    TagData,
)
from lightly.openapi_generated.swagger_client.rest import ApiException

STATE_SCHEDULED_ID_NOT_FOUND = "CANCELED_OR_NOT_EXISTING"


class InvalidConfigurationError(RuntimeError):
    pass


@dataclasses.dataclass
class ComputeWorkerRunInfo:
    """Information about a Lightly Worker run.

    Attributes:
        state:
            The state of the Lightly Worker run.
        message:
            The last message of the Lightly Worker run.
    """

    state: Union[
        DockerRunState, DockerRunScheduledState.OPEN, STATE_SCHEDULED_ID_NOT_FOUND
    ]
    message: str

    def in_end_state(self) -> bool:
        """Checks whether the Lightly Worker run has ended."""
        return self.state in [
            DockerRunState.COMPLETED,
            DockerRunState.ABORTED,
            DockerRunState.FAILED,
            DockerRunState.CRASHED,
            STATE_SCHEDULED_ID_NOT_FOUND,
        ]

    def ended_successfully(self) -> bool:
        """Checkes whether the Lightly Worker run ended successfully or failed.

        Returns:
            A boolean value indicating if the Lightly Worker run was successful.
            True if the run was successful.

        Raises:
            ValueError:
                If the Lightly Worker run is still in progress.
        """
        if not self.in_end_state():
            raise ValueError("Lightly Worker run is still in progress.")
        return self.state == DockerRunState.COMPLETED


class _ComputeWorkerMixin:
    def register_compute_worker(
        self, name: str = "Default", labels: Optional[List[str]] = None
    ) -> str:
        """Registers a new Lightly Worker.

        The ID of the registered worker will be returned. If a worker with the same
        name already exists, the ID of the existing worker is returned.

        Args:
            name:
                The name of the Lightly Worker.
            labels:
                The labels of the Lightly Worker.
                See our docs for more information regarding the labels parameter:
                https://docs.lightly.ai/docs/assign-scheduled-runs-to-specific-workers

        Returns:
            ID of the registered Lightly Worker.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> worker_id = client.register_compute_worker(name="my-worker", labels=["worker-label"])
            >>> worker_id
            '64709eac61e9ce68180a6529'
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
        """Fetches the IDs of all registered Lightly Workers.

        Returns:
            A list of worker IDs.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> worker_ids = client.get_compute_worker_ids()
            >>> worker_ids
            ['64709eac61e9ce68180a6529', '64709f8f61e9ce68180a652a']
        """
        entries = self._compute_worker_api.get_docker_worker_registry_entries()
        return [entry.id for entry in entries]

    def get_compute_workers(self) -> List[DockerWorkerRegistryEntryData]:
        """Fetches details of all registered Lightly Workers.

        Returns:
            A list of Lightly Worker details.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> workers = client.get_compute_workers()
            >>> workers
            [{'created_at': 1685102336056,
                'docker_version': '2.6.0',
                'id': '64709eac61e9ce68180a6529',
                'labels': [],
                ...
            }]
        """
        entries: list[
            DockerWorkerRegistryEntryData
        ] = self._compute_worker_api.get_docker_worker_registry_entries()
        return entries

    def delete_compute_worker(self, worker_id: str) -> None:
        """Removes a Lightly Worker.

        Args:
            worker_id:
                ID of the worker to be removed.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> worker_ids = client.get_compute_worker_ids()
            >>> worker_ids
            ['64709eac61e9ce68180a6529']
            >>> client.delete_compute_worker(worker_id="64709eac61e9ce68180a6529")
            >>> client.get_compute_worker_ids()
            []
        """
        self._compute_worker_api.delete_docker_worker_registry_entry_by_id(worker_id)

    def create_compute_worker_config(
        self,
        worker_config: Optional[Dict[str, Any]] = None,
        lightly_config: Optional[Dict[str, Any]] = None,
        selection_config: Optional[Union[Dict[str, Any], SelectionConfigV4]] = None,
    ) -> str:
        """Creates a new configuration for a Lightly Worker run.

        See our docs for more information regarding the different configurations:
        https://docs.lightly.ai/docs/all-configuration-options

        Args:
            worker_config:
                Lightly Worker configuration.
            lightly_config:
                Lightly configuration.
            selection_config:
                Selection configuration.

        Returns:
            The ID of the created config.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> selection_config = {
            ...     "n_samples": 3,
            ...     "strategies": [
            ...         {
            ...             "input": {"type": "RANDOM", "random_seed": 42},
            ...             "strategy": {"type": "WEIGHTS"},
            ...         }
            ...     ],
            ... }
            >>> config_id = client.create_compute_worker_config(
            ...     selection_config=selection_config,
            ... )

        :meta private:  # Skip docstring generation
        """
        if isinstance(selection_config, dict):
            selection = selection_config_from_dict(cfg=selection_config)
        else:
            selection = selection_config

        if worker_config is not None:
            worker_config_cc = _config_to_camel_case(cfg=worker_config)
            deserialize_worker_config = _get_deserialize(
                api_client=self.api_client,
                klass=DockerWorkerConfigV4Docker,
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

        config = DockerWorkerConfigV4(
            worker_type=DockerWorkerType.FULL,
            docker=docker,
            lightly=lightly,
            selection=selection,
        )
        request = DockerWorkerConfigOmniVXCreateRequest.from_dict(
            {
                "version": "V4",
                "config": config.to_dict(by_alias=True),
                "creator": self._creator,
            }
        )
        response = self._compute_worker_api.create_docker_worker_config_vx(request)
        return response.id

    def schedule_compute_worker_run(
        self,
        worker_config: Optional[Dict[str, Any]] = None,
        lightly_config: Optional[Dict[str, Any]] = None,
        selection_config: Optional[Union[Dict[str, Any], SelectionConfigV4]] = None,
        priority: str = DockerRunScheduledPriority.MID,
        runs_on: Optional[List[str]] = None,
    ) -> str:
        """Schedules a run with the given configurations.

        See our docs for more information regarding the different configurations:
        https://docs.lightly.ai/docs/all-configuration-options

        Args:
            worker_config:
                Lightly Worker configuration.
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

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> selection_config = {...}
            >>> worker_labels = ["worker-label"]
            >>> run_id = client.schedule_compute_worker_run(
            ...     selection_config=selection_config, runs_on=worker_labels
            ... )
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
            docker_run_scheduled_create_request=request,
            dataset_id=self.dataset_id,
        )
        return response.id

    def get_compute_worker_runs_iter(
        self,
        dataset_id: Optional[str] = None,
    ) -> Iterator[DockerRunData]:
        """Returns an iterator over all Lightly Worker runs for the user.

        Args:
            dataset_id:
                Target dataset ID. Optional. If set, only runs with the given dataset
                will be returned.

        Returns:
            Runs iterator.

        """
        if dataset_id is not None:
            return utils.paginate_endpoint(
                self._compute_worker_api.get_docker_runs_query_by_dataset_id,
                dataset_id=dataset_id,
            )
        else:
            return utils.paginate_endpoint(
                self._compute_worker_api.get_docker_runs,
            )

    def get_compute_worker_runs(
        self,
        dataset_id: Optional[str] = None,
    ) -> List[DockerRunData]:
        """Fetches all Lightly Worker runs for the user.

        Args:
            dataset_id:
                Target dataset ID. Optional. If set, only runs with the given dataset
                will be returned.

        Returns:
            Runs sorted by creation time from the oldest to the latest.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> client.get_compute_worker_runs()
            [{'artifacts': [...],
             'config_id': '6470a16461e9ce68180a6530',
             'created_at': 1679479418110,
             'dataset_id': '6470a36361e9ce68180a6531',
             'docker_version': '2.6.0',
             ...
             }]
        """
        runs: List[DockerRunData] = list(self.get_compute_worker_runs_iter(dataset_id))
        sorted_runs = sorted(runs, key=lambda run: run.created_at or -1)
        return sorted_runs

    def get_compute_worker_run(self, run_id: str) -> DockerRunData:
        """Fetches a Lightly Worker run.

        Args:
            run_id: Run ID.

        Returns:
            Details of the Lightly Worker run.

        Raises:
            ApiException:
                If no run with the given ID exists.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> client.get_compute_worker_run(run_id="6470a20461e9ce68180a6530")
            {'artifacts': [...],
             'config_id': '6470a16461e9ce68180a6530',
             'created_at': 1679479418110,
             'dataset_id': '6470a36361e9ce68180a6531',
             'docker_version': '2.6.0',
             ...
             }
        """
        return self._compute_worker_api.get_docker_run_by_id(run_id=run_id)

    def get_compute_worker_run_from_scheduled_run(
        self,
        scheduled_run_id: str,
    ) -> DockerRunData:
        """Fetches a Lightly Worker run given its scheduled run ID.

        Args:
            scheduled_run_id: Scheduled run ID.

        Returns:
            Details of the Lightly Worker run.

        Raises:
            ApiException:
                If no run with the given scheduled run ID exists or if the scheduled
                run is not yet picked up by a worker.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> client.get_compute_worker_run_from_scheduled_run(scheduled_run_id="646f338a8a5613b57d8b73a1")
            {'artifacts': [...],
             'config_id': '6470a16461e9ce68180a6530',
             'created_at': 1679479418110,
             'dataset_id': '6470a36361e9ce68180a6531',
             'docker_version': '2.6.0',
             ...
            }
        """
        return self._compute_worker_api.get_docker_run_by_scheduled_id(
            scheduled_id=scheduled_run_id
        )

    def get_scheduled_compute_worker_runs(
        self,
        state: Optional[str] = None,
    ) -> List[DockerRunScheduledData]:
        """Returns a list of scheduled Lightly Worker runs with the current dataset.

        Args:
            state:
                DockerRunScheduledState value. If specified, then only runs in the given
                state are returned. If omitted, then runs which have not yet finished
                (neither 'DONE' nor 'CANCELED') are returned. Valid states are 'OPEN',
                'LOCKED', 'DONE', and 'CANCELED'.

        Returns:
            A list of scheduled Lightly Worker runs.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>> client.get_scheduled_compute_worker_runs(state="OPEN")
            [{'config_id': '646f34608a5613b57d8b73cc',
             'created_at': 1685009508254,
             'dataset_id': '6470a36361e9ce68180a6531',
             'id': '646f338a8a5613b57d8b73a1',
             'last_modified_at': 1685009542667,
             'owner': '643d050b8bcb91967ded65df',
             'priority': 'MID',
             'runs_on': ['worker-label'],
             'state': 'OPEN'}]
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
                The ID with which the run was scheduled.

        Returns:
            Defails of the scheduled run.

        """
        try:
            run: DockerRunScheduledData = next(
                run
                for run in utils.retry(
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
        """Returns information about the Lightly Worker run.

        Args:
            scheduled_run_id:
                ID of the scheduled run.

        Returns:
            Details of the Lightly Worker run.

        Examples:
            >>> # Scheduled a Lightly Worker run and get its state
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
        """Pulls information about a Lightly Worker run continuously.

        Polls the Lightly Worker status every 30s.
        If the status changed, an update pops up.
        If the Lightly Worker run finished, the generator stops.

        Args:
            scheduled_run_id:
                The id with which the run was scheduled.

        Returns:
            Generator of information about the Lightly Worker run status.

        Examples:
            >>> # Scheduled a Lightly Worker run and monitor its state
            >>> scheduled_run_id = client.schedule_compute_worker_run(...)
            >>> for run_info in client.compute_worker_run_info_generator(scheduled_run_id):
            >>>     print(f"Lightly Worker run is now in state='{run_info.state}' with message='{run_info.message}'")
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
        """Returns all tags from a run with the current dataset.

        Only returns tags for runs made with Lightly Worker version >=2.4.2.

        Args:
            run_id:
                Run ID from which to return tags.

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


def selection_config_from_dict(cfg: Dict[str, Any]) -> SelectionConfigV4:
    """Recursively converts selection config from dict to a SelectionConfigV4 instance."""
    new_cfg = copy.deepcopy(cfg)
    strategies = []
    for entry in new_cfg.get("strategies", []):
        strategies.append(selection_config_entry_from_dict(entry=entry))
    new_cfg["strategies"] = strategies
    auto_tasks = []
    for entry in new_cfg.get("auto_tasks", []):
        auto_tasks.append(auto_task_from_dict(entry=entry))
    new_cfg["auto_tasks"] = auto_tasks
    return SelectionConfigV4(**new_cfg)


def selection_config_entry_from_dict(entry: Dict[str, Any]) -> AutoTask:
    new_entry = copy.deepcopy(entry)
    new_entry["input"] = SelectionConfigV4EntryInput(**new_entry["input"])
    new_entry["strategy"] = SelectionConfigV4EntryStrategy(**new_entry["strategy"])
    return SelectionConfigV4Entry(**new_entry)


def auto_task_from_dict(entry: Dict[str, Any]) -> AutoTask:
    auto_task_type_to_class = {
        "TILING": AutoTaskTiling,
    }
    if entry["type"] not in auto_task_type_to_class:
        raise ValueError(
            f"AutoTask type '{entry['type']}' not supported. "
            f"Supported types are: {list(auto_task_type_to_class.keys())}"
        )
    auto_task_class = auto_task_type_to_class[entry["type"]]
    auto_task_instance = auto_task_class(**entry)
    return AutoTask(actual_instance=auto_task_instance)


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
    suggests a best match.

    Raises:
        InvalidConfigurationError: If obj is not a valid config.

    """

    if cfg is None:
        return

    for key, item in cfg.items():
        if not hasattr(obj, key):
            possible_options = list(obj.__fields__.keys())
            closest_match = difflib.get_close_matches(
                word=key, possibilities=possible_options, n=1, cutoff=0.0
            )[0]
            error_msg = (
                f"Option '{key}' does not exist! Did you mean '{closest_match}'?"
            )
            raise InvalidConfigurationError(error_msg)
        if isinstance(item, dict):
            _validate_config(item, getattr(obj, key))
