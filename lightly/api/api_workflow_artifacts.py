import os
import warnings

from lightly.api import download
from lightly.openapi_generated.swagger_client.models import (
    DockerRunArtifactData,
    DockerRunArtifactType,
    DockerRunData,
)


class ArtifactNotExist(Exception):
    pass


class _ArtifactsMixin:
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
            >>> run = client.get_compute_worker_run_from_scheduled_run(scheduled_run_id=scheduled_run_id)
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
            >>> run = client.get_compute_worker_run_from_scheduled_run(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_checkpoint(run=run, output_path="my_checkpoint.ckpt")

        """
        self._download_compute_worker_run_artifact_by_type(
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
            >>> run = client.get_compute_worker_run_from_scheduled_run(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_report_pdf(run=run, output_path="report.pdf")

        """
        self._download_compute_worker_run_artifact_by_type(
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

        DEPRECATED: This method is deprecated and will be removed in the future. Use
        download_compute_worker_run_report_v2_json to download the new report_v2.json
        instead.

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
            >>> run = client.get_compute_worker_run_from_scheduled_run(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_report_json(run=run, output_path="report.json")

        """
        warnings.warn(
            DeprecationWarning(
                "This method downloads the deprecated report.json file and will be "
                "removed in the future. Use download_compute_worker_run_report_v2_json "
                "to download the new report_v2.json file instead."
            )
        )
        self._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=DockerRunArtifactType.REPORT_JSON,
            output_path=output_path,
            timeout=timeout,
        )

    def download_compute_worker_run_report_v2_json(
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
            >>> run = client.get_compute_worker_run_from_scheduled_run(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_report_v2_json(run=run, output_path="report_v2.json")

        """
        self._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=DockerRunArtifactType.REPORT_V2_JSON,
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
            >>> run = client.get_compute_worker_run_from_scheduled_run(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_log(run=run, output_path="log.txt")

        """
        self._download_compute_worker_run_artifact_by_type(
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
            >>> run = client.get_compute_worker_run_from_scheduled_run(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_memory_log(run=run, output_path="memlog.txt")

        """
        self._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=DockerRunArtifactType.MEMLOG,
            output_path=output_path,
            timeout=timeout,
        )

    def download_compute_worker_run_corruptness_check_information(
        self,
        run: DockerRunData,
        output_path: str,
        timeout: int = 60,
    ) -> None:
        """Download the corruptness check information file from a run.

        Args:
            run:
                Run from which to download the file.
            output_path:
                Path where the file will be saved.
            timeout:
                Timeout in seconds after which download is interrupted.

        Raises:
            ArtifactNotExist:
                If the run has no corruptness check information artifact or the file
                has not yet been uploaded.

        Examples:
            >>> # schedule run
            >>> scheduled_run_id = client.schedule_compute_worker_run(...)
            >>>
            >>> # wait until run completed
            >>> for run_info in client.compute_worker_run_info_generator(scheduled_run_id=scheduled_run_id):
            >>>     pass
            >>>
            >>> # download corruptness check information file
            >>> run = client.get_compute_worker_run_from_scheduled_run(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_corruptness_check_information(run=run, output_path="corruptness_check_information.json")
            >>>
            >>> # print all corrupt samples and corruptions
            >>> with open("corruptness_check_information.json", 'r') as f:
            >>>     corruptness_check_information = json.load(f)
            >>> for sample_name, error in corruptness_check_information["corrupt_samples"].items():
            >>>     print(f"Sample '{sample_name}' is corrupt because of the error '{error}'.")

        """
        self._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=DockerRunArtifactType.CORRUPTNESS_CHECK_INFORMATION,
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
            >>> run = client.get_compute_worker_run_from_scheduled_run(scheduled_run_id=scheduled_run_id)
            >>> client.download_compute_worker_run_sequence_information(run=run, output_path="sequence_information.json")

        """
        self._download_compute_worker_run_artifact_by_type(
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
        artifact = self._get_artifact_by_type(artifact_type, run)
        self._download_compute_worker_run_artifact(
            run_id=run.id,
            artifact_id=artifact.id,
            output_path=output_path,
            timeout=timeout,
        )

    def _get_artifact_by_type(
        self, artifact_type: str, run: DockerRunData
    ) -> DockerRunArtifactData:
        if run.artifacts is None:
            raise ArtifactNotExist(f"Run has no artifacts.")
        try:
            artifact = next(art for art in run.artifacts if art.type == artifact_type)
        except StopIteration:
            raise ArtifactNotExist(
                f"No artifact with type '{artifact_type}' in artifacts."
            )
        return artifact

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

    def get_compute_worker_run_checkpoint_url(
        self,
        run: DockerRunData,
    ) -> str:
        """Gets the download url of the last training checkpoint from a run.

        See our docs for more information regarding checkpoints:
        https://docs.lightly.ai/docs/train-a-self-supervised-model#checkpoints

        Args:
            run:
                Run from which to download the checkpoint.

        Returns:
            The url from which the checkpoint can be downloaded.

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
            >>> # get checkpoint read_url
            >>> run = client.get_compute_worker_run_from_scheduled_run(scheduled_run_id=scheduled_run_id)
            >>> checkpoint_read_url = client.get_compute_worker_run_checkpoint_url(run=run)

        """
        artifact = self._get_artifact_by_type(
            artifact_type=DockerRunArtifactType.CHECKPOINT, run=run
        )
        read_url = self._compute_worker_api.get_docker_run_artifact_read_url_by_id(
            run_id=run.id, artifact_id=artifact.id
        )
        return read_url
