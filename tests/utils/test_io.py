import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lightly.utils import io
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


class TestCLICrop(MockedApiWorkflowSetup):  # type: ignore[misc]
    def test_save_metadata(self) -> None:
        metadata = [("filename.jpg", {"random_metadata": 42})]
        metadata_filepath = tempfile.mktemp(".json", "metadata")
        io.save_custom_metadata(metadata_filepath, metadata)


class TestEmbeddingsIO(unittest.TestCase):
    def setUp(self) -> None:
        # correct embedding file as created through lightly
        self.embeddings_path = tempfile.mktemp(".csv", "embeddings")
        embeddings = np.random.rand(32, 2)
        labels = [0 for i in range(len(embeddings))]
        filenames = [f"img_{i}.jpg" for i in range(len(embeddings))]
        io.save_embeddings(self.embeddings_path, embeddings, labels, filenames)

    def test_valid_embeddings(self) -> None:
        io.check_embeddings(self.embeddings_path)

    def test_whitespace_in_embeddings(self) -> None:
        # should fail because there whitespaces in the header columns
        lines = [
            "filenames, embedding_0,embedding_1,labels\n",
            "img_1.jpg, 0.351,0.1231",
        ]
        with open(self.embeddings_path, "w") as f:
            f.writelines(lines)
        with self.assertRaises(RuntimeError) as context:
            io.check_embeddings(self.embeddings_path)
        self.assertTrue("must not contain whitespaces" in str(context.exception))

    def test_no_labels_in_embeddings(self) -> None:
        # should fail because there is no `labels` column in the header
        lines = ["filenames,embedding_0,embedding_1\n", "img_1.jpg,0.351,0.1231"]
        with open(self.embeddings_path, "w") as f:
            f.writelines(lines)
        with self.assertRaises(RuntimeError) as context:
            io.check_embeddings(self.embeddings_path)
        self.assertTrue("has no `labels` column" in str(context.exception))

    def test_no_empty_rows_in_embeddings(self) -> None:
        # should fail because there are empty rows in the embeddings file
        lines = [
            "filenames,embedding_0,embedding_1,labels\n",
            "img_1.jpg,0.351,0.1231\n\n" "img_2.jpg,0.311,0.6231",
        ]
        with open(self.embeddings_path, "w") as f:
            f.writelines(lines)
        with self.assertRaises(RuntimeError) as context:
            io.check_embeddings(self.embeddings_path)
        self.assertTrue("must not have empty rows" in str(context.exception))

    def test_embeddings_extra_rows(self) -> None:
        rows = [
            ["filenames", "embedding_0", "embedding_1", "labels", "selected", "masked"],
            ["image_0.jpg", "3.4", "0.23", "0", "1", "0"],
            ["image_1.jpg", "3.4", "0.23", "1", "0", "1"],
        ]
        with open(self.embeddings_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(rows)

        io.check_embeddings(self.embeddings_path, remove_additional_columns=True)

        with open(self.embeddings_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row_read, row_original in zip(csv_reader, rows):
                self.assertListEqual(row_read, row_original[:-2])

    def test_embeddings_extra_rows_special_order(self) -> None:
        input_rows = [
            ["filenames", "embedding_0", "embedding_1", "masked", "labels", "selected"],
            ["image_0.jpg", "3.4", "0.23", "0", "1", "0"],
            ["image_1.jpg", "3.4", "0.23", "1", "0", "1"],
        ]
        correct_output_rows = [
            ["filenames", "embedding_0", "embedding_1", "labels"],
            ["image_0.jpg", "3.4", "0.23", "1"],
            ["image_1.jpg", "3.4", "0.23", "0"],
        ]
        with open(self.embeddings_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(input_rows)

        io.check_embeddings(self.embeddings_path, remove_additional_columns=True)

        with open(self.embeddings_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row_read, row_original in zip(csv_reader, correct_output_rows):
                self.assertListEqual(row_read, row_original)

    def test_save_tasks(self) -> None:
        tasks = [
            "task1",
            "task2",
            "task3",
        ]
        with tempfile.NamedTemporaryFile(suffix=".json") as file:
            io.save_tasks(file.name, tasks)
            with open(file.name, "r") as f:
                loaded = json.load(f)
        self.assertListEqual(tasks, loaded)

    def test_save_schema(self) -> None:
        description = "classification"
        ids = [1, 2, 3, 4]
        names = ["name1", "name2", "name3", "name4"]
        expected_format = {
            "task_type": "classification",
            "categories": [
                {"id": 1, "name": "name1"},
                {"id": 2, "name": "name2"},
                {"id": 3, "name": "name3"},
                {"id": 4, "name": "name4"},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".json") as file:
            io.save_schema(file.name, description, ids, names)
            with open(file.name, "r") as f:
                loaded = json.load(f)
        self.assertListEqual(sorted(expected_format), sorted(loaded))

    def test_save_schema_different(self) -> None:
        with self.assertRaises(ValueError):
            io.save_schema(
                "name_doesnt_matter",
                "description_doesnt_matter",
                [1, 2],
                ["name1"],
            )


def test_save_and_load_embeddings(tmp_path: Path) -> None:
    embeddings = np.random.rand(2, 32)
    labels = [0, 1]
    filenames = ["img_1.jpg", "img_2.jpg"]

    io.save_embeddings(
        path=str(tmp_path / "embeddings.csv"),
        embeddings=embeddings,
        labels=labels,
        filenames=filenames,
    )

    loaded_embeddings, loaded_labels, loaded_filenames = io.load_embeddings(
        path=str(tmp_path / "embeddings.csv")
    )
    assert np.allclose(embeddings, loaded_embeddings)
    assert labels == loaded_labels
    assert filenames == loaded_filenames


def test_save_and_load_embeddings__filename_with_comma(tmp_path: Path) -> None:
    embeddings = np.random.rand(4, 32)
    labels = [0, 1, 2, 3]
    filenames = ["img,1.jpg", '",img,.jpg', ',"img".jpg', ',"img\n".jpg']

    io.save_embeddings(
        path=str(tmp_path / "embeddings.csv"),
        embeddings=embeddings,
        labels=labels,
        filenames=filenames,
    )

    loaded_embeddings, loaded_labels, loaded_filenames = io.load_embeddings(
        path=str(tmp_path / "embeddings.csv")
    )
    assert np.allclose(embeddings, loaded_embeddings)
    assert labels == loaded_labels
    assert filenames == loaded_filenames
