import copy
import dataclasses
import os
import time
from typing import Any, Dict, List, Optional, Union, Iterator

from lightly.api.utils import retry
from lightly.api import download
from lightly.api import utils
from lightly.openapi_generated.swagger_client import (
    CreateDockerWorkerRegistryEntryRequest,
    DockerRunData,
    DockerRunScheduledCreateRequest,
    DockerRunScheduledData,
    DockerRunScheduledPriority,
    DockerRunScheduledState,
    DockerRunState,
    DockerWorkerConfig,
    DockerWorkerConfigCreateRequest,
    DockerWorkerType,
    SelectionConfig,
    SelectionConfigEntry,
    SelectionConfigEntryInput,
    SelectionConfigEntryStrategy,
)
from lightly.openapi_generated.swagger_client.models.docker_run_artifact_type import DockerRunArtifactType
from lightly.openapi_generated.swagger_client.rest import ApiException

STATE_SCHEDULED_ID_NOT_FOUND = "CANCELED_OR_NOT_EXISTING"

class ArtifactNotExist(Exception):
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
    def register_compute_worker(self, name: str = "Default", labels: Optional[List[str]] = None) -> str:
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
            name=name, worker_type=DockerWorkerType.FULL, labels=labels
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
        runs_on: Optional[List[str]] = None
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

        """
        if runs_on is None:
            runs_on = []
        config_id = self.create_compute_worker_config(
            worker_config=worker_config,
            lightly_config=lightly_config,
            selection_config=selection_config,
        )
        request = DockerRunScheduledCreateRequest(
            config_id=config_id, priority=priority, runs_on=runs_on
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
                dataset_id=dataset_id
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
        return self._compute_worker_api.get_docker_run_by_id(
            run_id=run_id
        )

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
                dataset_id=self.dataset_id, state=state,
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

    def download_compute_worker_run_artifacts(
        self,
        run: DockerRunData,
        output_dir: str,
        timeout: int = 60,
    ) -> None:
        """Downloads all artifacts from a run.
        
        Args:
            run:
                Run from which to download artifacts.
            output_dir:
                Output directory where artifacts will be saved.
            timeout:
                Timeout in seconds after which an artifact download is interrupted.

        Examples:
            >>> # schedule run
            >>> scheduled_run_id = client.schedule_compute_worker_run(...)
            >>>
            >>> # wait until run completed
            >>> for run_info in client.compute_worker_run_info_generator(scheduled_run_id=scheduled_run_id):
            >>>     pass
            >>>
            >>> # download artifacts
            >>> run = client.get_compute_worker_run_from_scheduled(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_artifacts(run=run, output_dir="my_run/artifacts")

        """
        if run.artifacts is None:
            return
        for artifact in run.artifacts:
            self._download_compute_worker_run_artifact(
                run_id=run.id,
                artifact_id=artifact.id,
                output_path=os.path.join(output_dir, artifact.file_name),
                timeout=timeout,
            )

    def download_compute_worker_run_checkpoint(
        self,
        run: DockerRunData,
        output_path: str,
        timeout: int = 60,
    ) -> None:
        """Downloads the last training checkpoint from a run.

        See our docs for more information regarding checkpoints:
        https://docs.lightly.ai/docs/train-a-self-supervised-model#checkpoints

        Args:
            run:
                Run from which to download the checkpoint.
            output_path:
                Path where checkpoint will be saved.
            timeout:
                Timeout in seconds after which download is interrupted.

        Raises:
            ArtifactNotExist:
                If the run has no checkpoint artifact or the checkpoint has not yet been
                uploaded.

        Examples:
            >>> # schedule run
            >>> scheduled_run_id = client.schedule_compute_worker_run(...)
            >>>
            >>> # wait until run completed
            >>> for run_info in client.compute_worker_run_info_generator(scheduled_run_id=scheduled_run_id):
            >>>     pass
            >>>
            >>> # download checkpoint
            >>> run = client.get_compute_worker_run_from_scheduled(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_checkpoint(run=run, output_path="my_checkpoint.ckpt")

        """
        return self._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=DockerRunArtifactType.CHECKPOINT,
            output_path=output_path,
            timeout=timeout,
        )

    def download_compute_worker_run_report_pdf(
        self, 
        run: DockerRunData, 
        output_path: str, 
        timeout: int = 60,
    ) -> None:
        """Download the report in pdf format from a run.

        Args:
            run:
                Run from which to download the report.
            output_path:
                Path where report will be saved.
            timeout:
                Timeout in seconds after which download is interrupted.

        Raises:
            ArtifactNotExist:
                If the run has no report artifact or the report has not yet been
                uploaded.

        Examples:
            >>> # schedule run
            >>> scheduled_run_id = client.schedule_compute_worker_run(...)
            >>>
            >>> # wait until run completed
            >>> for run_info in client.compute_worker_run_info_generator(scheduled_run_id=scheduled_run_id):
            >>>     pass
            >>>
            >>> # download report
            >>> run = client.get_compute_worker_run_from_scheduled(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_report_pdf(run=run, output_path="report.pdf")

        """
        return self._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=DockerRunArtifactType.REPORT_PDF,
            output_path=output_path,
            timeout=timeout,
        )

    def download_compute_worker_run_report_json(
        self,
        run: DockerRunData,
        output_path: str,
        timeout: int = 60,
    ) -> None:
        """Download the report in json format from a run.

        Args:
            run:
                Run from which to download the report.
            output_path:
                Path where report will be saved.
            timeout:
                Timeout in seconds after which download is interrupted.

        Raises:
            ArtifactNotExist:
                If the run has no report artifact or the report has not yet been
                uploaded.

        Examples:
            >>> # schedule run
            >>> scheduled_run_id = client.schedule_compute_worker_run(...)
            >>>
            >>> # wait until run completed
            >>> for run_info in client.compute_worker_run_info_generator(scheduled_run_id=scheduled_run_id):
            >>>     pass
            >>>
            >>> # download checkpoint
            >>> run = client.get_compute_worker_run_from_scheduled(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_report_json(run=run, output_path="report.json")

        """
        return self._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=DockerRunArtifactType.REPORT_JSON,
            output_path=output_path,
            timeout=timeout,
        )

    def download_compute_worker_run_log(
        self,
        run: DockerRunData,
        output_path: str,
        timeout: int = 60,
    ) -> None:
        """Download the log file from a run.

        Args:
            run:
                Run from which to download the log file.
            output_path:
                Path where log file will be saved.
            timeout:
                Timeout in seconds after which download is interrupted.

        Raises:
            ArtifactNotExist:
                If the run has no log artifact or the log file has not yet been 
                uploaded.

        Examples:
            >>> # schedule run
            >>> scheduled_run_id = client.schedule_compute_worker_run(...)
            >>>
            >>> # wait until run completed
            >>> for run_info in client.compute_worker_run_info_generator(scheduled_run_id=scheduled_run_id):
            >>>     pass
            >>>
            >>> # download log file
            >>> run = client.get_compute_worker_run_from_scheduled(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_log(run=run, output_path="log.txt")

        """
        return self._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=DockerRunArtifactType.LOG,
            output_path=output_path,
            timeout=timeout,
        )

    def download_compute_worker_run_memory_log(
        self,
        run: DockerRunData,
        output_path: str,
        timeout: int = 60,
    ) -> None:
        """Download the memory consumption log file from a run.

        Args:
            run:
                Run from which to download the memory log file.
            output_path:
                Path where memory log file will be saved.
            timeout:
                Timeout in seconds after which download is interrupted.

        Raises:
            ArtifactNotExist:
                If the run has no memory log artifact or the memory log file has not yet
                been uploaded.

        Examples:
            >>> # schedule run
            >>> scheduled_run_id = client.schedule_compute_worker_run(...)
            >>>
            >>> # wait until run completed
            >>> for run_info in client.compute_worker_run_info_generator(scheduled_run_id=scheduled_run_id):
            >>>     pass
            >>>
            >>> # download memory log file
            >>> run = client.get_compute_worker_run_from_scheduled(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_memory_log(run=run, output_path="memlog.txt")

        """
        return self._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=DockerRunArtifactType.MEMLOG,
            output_path=output_path,
            timeout=timeout,
        )

    def download_compute_worker_run_sequence_information(
        self,
        run: DockerRunData,
        output_path: str,
        timeout: int = 60,
    ) -> None:
        """Download the sequence information from a run.

        Args:
            run:
                Run from which to download the the file.
            output_path:
                Path where the file will be saved.
            timeout:
                Timeout in seconds after which download is interrupted.

        Raises:
            ArtifactNotExist:
                If the run has no sequence information artifact or the file has not yet
                been uploaded.

        Examples:
            >>> # schedule run
            >>> scheduled_run_id = client.schedule_compute_worker_run(...)
            >>>
            >>> # wait until run completed
            >>> for run_info in client.compute_worker_run_info_generator(scheduled_run_id=scheduled_run_id):
            >>>     pass
            >>>
            >>> # download sequence information file
            >>> run = client.get_compute_worker_run_from_scheduled(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_sequence_information(run=run, output_path="memlog.txt")

        """
        return self._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=DockerRunArtifactType.SEQUENCE_INFORMATION,
            output_path=output_path,
            timeout=timeout,
        )

    def _download_compute_worker_run_artifact_by_type(
        self,
        run: DockerRunData,
        artifact_type: str,
        output_path: str,
        timeout: int,
    ) -> None:
        if run.artifacts is None:
            raise ArtifactNotExist(f"Run has no artifacts.")
        try:
            artifact = next(art for art in run.artifacts if art.type == artifact_type)
        except StopIteration:
            raise ArtifactNotExist(f"No artifact with type '{artifact_type}' in artifacts.")
        self._download_compute_worker_run_artifact(
            run_id=run.id,
            artifact_id=artifact.id,
            output_path=output_path,
            timeout=timeout,
        )

    def _download_compute_worker_run_artifact(
        self,
        run_id: str,
        artifact_id: str,
        output_path: str,
        timeout: int,
    ) -> None:
        read_url = self._compute_worker_api.get_docker_run_artifact_read_url_by_id(
            run_id=run_id,
            artifact_id=artifact_id,
        )
        download.download_and_write_file(
            url=read_url,
            output_path=output_path,
            request_kwargs=dict(timeout=timeout),
        )

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
