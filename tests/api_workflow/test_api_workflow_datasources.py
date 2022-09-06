from unittest import mock

import tqdm
import pytest

from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data_row import (
    DatasourceRawSamplesDataRow,
)
from collections import defaultdict


class TestApiWorkflowDatasources(MockedApiWorkflowSetup):
    def test_get_processed_until_timestamp(self):
        self.api_workflow_client._datasources_api.reset()
        assert self.api_workflow_client.get_processed_until_timestamp() == 0

    def test_update_processed_until_timestamp(self):
        self.api_workflow_client._datasources_api.reset()
        self.api_workflow_client.update_processed_until_timestamp(10)
        assert self.api_workflow_client.get_processed_until_timestamp() == 10

    def test_download_raw_samples(self):
        self.api_workflow_client._datasources_api.reset()
        samples = self.api_workflow_client.download_raw_samples()
        num_samples = self.api_workflow_client._datasources_api._num_samples
        assert len(samples) == num_samples

    def test_download_raw_samples_progress_bar(self):
        self.api_workflow_client._datasources_api.reset()
        pbar = mock.Mock(wraps=tqdm.tqdm(unit="file"))
        samples = self.api_workflow_client.download_raw_samples(progress_bar=pbar)
        num_samples = self.api_workflow_client._datasources_api._num_samples
        assert len(samples) == num_samples
        pbar.update.assert_called()

    def test_download_raw_samples_no_duplicates(self):
        self.api_workflow_client._datasources_api.reset()
        samples = self.api_workflow_client.download_raw_samples()
        assert len(samples) == len(set(samples))

    def test_download_new_raw_samples_no_duplicates(self):
        self.api_workflow_client._datasources_api.reset()
        samples = self.api_workflow_client.download_new_raw_samples()
        assert len(samples) == len(set(samples))

    def test_download_new_raw_samples_not_yet_processed(self):
        self.api_workflow_client._datasources_api.reset()
        samples = self.api_workflow_client.download_raw_samples()
        num_samples = self.api_workflow_client._datasources_api._num_samples
        assert len(samples) == num_samples

    def test_download_new_raw_samples_partially_processed(self):
        self.api_workflow_client._datasources_api.reset()
        num_samples = self.api_workflow_client._datasources_api._num_samples
        n_processed = num_samples // 2
        n_remaining = num_samples - n_processed
        processed_timestamp = n_processed - 1
        self.api_workflow_client.update_processed_until_timestamp(processed_timestamp)
        samples = self.api_workflow_client.download_new_raw_samples()
        assert len(samples) == n_remaining

    def test_download_raw_samples_equal_to_download_all_raw_new_samples(self):
        self.api_workflow_client._datasources_api.reset()
        samples = self.api_workflow_client.download_raw_samples()
        new_samples = self.api_workflow_client.download_new_raw_samples()
        assert len(samples) == len(new_samples)
        assert set(samples) == set(new_samples)

    def test_download_raw_samples_or_metadata_relevant_filenames(self):
        self.api_workflow_client._datasources_api.reset()
        for method in [
            self.api_workflow_client.download_raw_samples,
            self.api_workflow_client.download_raw_metadata,
        ]:
            for relevant_filenames_path in [None, "", "relevant_filenames.txt"]:
                with self.subTest(
                    relevant_filenames_path=relevant_filenames_path, method=method
                ):
                    samples = method(
                        relevant_filenames_file_name=relevant_filenames_path
                    )
            with self.subTest(relevant_filenames_path="unset", method=method):
                samples = method()

    def test_set_azure_config(self):
        self.api_workflow_client.set_azure_config(
            container_name="my-container/name",
            account_name="my-account-name",
            sas_token="my-sas-token",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
        )

    def test_set_gcs_config(self):
        self.api_workflow_client.set_gcs_config(
            resource_path="gs://my-bucket/my-dataset",
            project_id="my-project-id",
            credentials="my-credentials",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
        )

    def test_set_local_config(self):
        self.api_workflow_client.set_local_config(
            resource_path="http://localhost:1234/path/to/my/data",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
        )

    def test_set_s3_config(self):
        self.api_workflow_client.set_s3_config(
            resource_path="s3://my-bucket/my-dataset",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
            region="eu-central-1",
            access_key="my-access-key",
            secret_access_key="my-secret-access-key",
        )

    def test_set_s3_delegated_access_config(self):
        self.api_workflow_client.set_s3_delegated_access_config(
            resource_path="s3://my-bucket/my-dataset",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
            region="eu-central-1",
            role_arn="my-role-arn",
            external_id="my-external-id",
        )

    def test_download_raw_samples_predictions(self):
        self.api_workflow_client._datasources_api.reset()

        predictions = self.api_workflow_client.download_raw_predictions("test")
        num_samples = self.api_workflow_client._datasources_api._num_samples
        assert len(predictions) == num_samples

    def test_download_raw_samples_predictions_progress_bar(self):
        self.api_workflow_client._datasources_api.reset()
        pbar = mock.Mock(wraps=tqdm.tqdm(unit="file"))
        predictions = self.api_workflow_client.download_raw_predictions(
            "test", progress_bar=pbar
        )
        num_samples = self.api_workflow_client._datasources_api._num_samples
        assert len(predictions) == num_samples
        pbar.update.assert_called()

    def test_download_raw_sample_metadata(self):
        self.api_workflow_client._datasources_api.reset()
        predictions = self.api_workflow_client.download_raw_metadata()
        num_samples = self.api_workflow_client._datasources_api._num_samples
        assert len(predictions) == num_samples

    def test_download_raw_sample_metadata_progress_bar(self):
        self.api_workflow_client._datasources_api.reset()
        pbar = mock.Mock(wraps=tqdm.tqdm(unit="file"))
        predictions = self.api_workflow_client.download_raw_metadata(progress_bar=pbar)
        num_samples = self.api_workflow_client._datasources_api._num_samples
        assert len(predictions) == num_samples
        pbar.update.assert_called()

    def test_download_raw_samples_predictions_relevant_filenames(self):
        self.api_workflow_client._datasources_api.reset()
        predictions = self.api_workflow_client.download_raw_predictions(
            "test", relevant_filenames_file_name="test"
        )
        num_samples = self.api_workflow_client._datasources_api._num_samples
        assert len(predictions) == num_samples

    def test_get_prediction_read_url(self):
        self.api_workflow_client._datasources_api.reset()
        read_url = self.api_workflow_client.get_prediction_read_url("test.json")
        self.assertIsNotNone(read_url)

    def test__download_raw_files_duplicate_filenames(self):
        self.api_workflow_client._datasources_api.reset()
        self.api_workflow_client._datasources_api._samples = defaultdict(
            lambda: [
                DatasourceRawSamplesDataRow(file_name="file_0", read_url="url_0"),
                DatasourceRawSamplesDataRow(file_name="file_1", read_url="url_1"),
                DatasourceRawSamplesDataRow(file_name="file_0", read_url="url_0"),
                DatasourceRawSamplesDataRow(file_name="file_2", read_url="url_2"),
                DatasourceRawSamplesDataRow(file_name="file_3", read_url="url_3"),
                DatasourceRawSamplesDataRow(file_name="file_4", read_url="url_4"),
            ]
        )
        with pytest.warns(
            UserWarning, match="Duplicate filename file_0 in relevant filenames file"
        ):
            samples = self.api_workflow_client.download_raw_samples()

        assert len(samples) == 5
        assert samples == [(f"file_{i}", f"url_{i}") for i in range(5)]

    def test__download_raw_files_absolute_filenames(self):
        self.api_workflow_client._datasources_api.reset
        self.api_workflow_client._datasources_api._samples = defaultdict(
            lambda: [
                DatasourceRawSamplesDataRow(file_name="/file_0", read_url="url_0"),
                DatasourceRawSamplesDataRow(file_name="file_1", read_url="url_1"),
                DatasourceRawSamplesDataRow(file_name="file_2", read_url="url_2"),
                DatasourceRawSamplesDataRow(file_name="file_3", read_url="url_3"),
                DatasourceRawSamplesDataRow(file_name="file_4", read_url="url_4"),
            ]
        )
        with pytest.warns(
            UserWarning,
            match="Absolute file paths like /file_0 are not supported in relevant filenames file",
        ):
            samples = self.api_workflow_client.download_raw_samples()

    def test__download_raw_files_dot_slash(self):
        self.api_workflow_client._datasources_api.reset
        self.api_workflow_client._datasources_api._samples = defaultdict(
            lambda: [
                DatasourceRawSamplesDataRow(file_name="./file_0", read_url="url_0"),
                DatasourceRawSamplesDataRow(file_name="file_1", read_url="url_1"),
                DatasourceRawSamplesDataRow(file_name="file_2", read_url="url_2"),
                DatasourceRawSamplesDataRow(file_name="file_3", read_url="url_3"),
                DatasourceRawSamplesDataRow(file_name="file_4", read_url="url_4"),
            ]
        )
        with pytest.warns(
            UserWarning,
            match="Using dot notation \('\./', '\.\./'\) like in \./file_0 is not supported.*",
        ):
            samples = self.api_workflow_client.download_raw_samples()

    def test__download_raw_files_dot_dot_slash(self):
        self.api_workflow_client._datasources_api.reset
        self.api_workflow_client._datasources_api._samples = defaultdict(
            lambda: [
                DatasourceRawSamplesDataRow(file_name="../file_0", read_url="url_0"),
                DatasourceRawSamplesDataRow(file_name="file_1", read_url="url_1"),
                DatasourceRawSamplesDataRow(file_name="file_2", read_url="url_2"),
                DatasourceRawSamplesDataRow(file_name="file_3", read_url="url_3"),
                DatasourceRawSamplesDataRow(file_name="file_4", read_url="url_4"),
            ]
        )
        with pytest.warns(
            UserWarning,
            match="Using dot notation \('\./', '\.\./'\) like in \.\./file_0 is not supported.*",
        ):
            samples = self.api_workflow_client.download_raw_samples()
