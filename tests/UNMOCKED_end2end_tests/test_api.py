import csv
import os
import sys
from typing import List, Tuple

import numpy as np
from hydra.experimental import compose, initialize

from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.cli import upload_cli
from lightly.data.dataset import LightlyDataset
from lightly.utils.io import save_embeddings


class CSVEmbeddingDataset:
    def __init__(self, path_to_embeddings_csv: str):
        with open(path_to_embeddings_csv, "r") as f:
            data = csv.reader(f)

            rows = list(data)
            header_row = rows[0]
            rows_without_header = rows[1:]

            index_filenames = header_row.index("filenames")
            filenames = [row[index_filenames] for row in rows_without_header]

            index_labels = header_row.index("labels")
            labels = [row[index_labels] for row in rows_without_header]

            embeddings = rows_without_header
            indexes_to_delete = sorted([index_filenames, index_labels], reverse=True)
            for embedding_row in embeddings:
                for index_to_delete in indexes_to_delete:
                    del embedding_row[index_to_delete]

        self.dataset = dict(
            [
                (filename, (np.array(embedding_row, dtype=float), int(label)))
                for filename, embedding_row, label in zip(filenames, embeddings, labels)
            ]
        )

    def get_features(self, filenames: List[str]) -> np.ndarray:
        features_array = np.array([self.dataset[filename][0] for filename in filenames])
        return features_array

    def get_labels(self, filenames: List[str]) -> np.ndarray:
        labels = np.array([self.dataset[filename][1] for filename in filenames])
        return labels

    def all_features_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        filenames = list(self.dataset.keys())
        features = self.get_features(filenames)
        labels = self.get_labels(filenames)
        return features, labels


def create_new_dataset_with_embeddings(
    path_to_dataset: str, token: str, dataset_name: str
) -> ApiWorkflowClient:
    api_workflow_client = ApiWorkflowClient(token=token)

    # create the dataset
    api_workflow_client.create_new_dataset_with_unique_name(
        dataset_basename=dataset_name
    )

    # upload to the dataset
    initialize(config_path="../../lightly/cli/config", job_name="test_app")
    cfg = compose(
        config_name="config",
        overrides=[
            f"input_dir='{path_to_dataset}'",
            f"token='{token}'",
            f"dataset_id={api_workflow_client.dataset_id}",
        ],
    )
    upload_cli(cfg)

    # calculate and save the embeddings
    path_to_embeddings_csv = f"{path_to_dataset}/embeddings.csv"
    if not os.path.isfile(path_to_embeddings_csv):
        dataset = LightlyDataset(input_dir=path_to_dataset)
        embeddings = np.random.normal(size=(len(dataset.dataset.samples), 32))
        filepaths, labels = zip(*dataset.dataset.samples)
        filenames = [
            filepath[len(path_to_dataset) :].lstrip("/") for filepath in filepaths
        ]
        print("Starting save of embeddings")
        save_embeddings(path_to_embeddings_csv, embeddings, labels, filenames)
        print("Finished save of embeddings")

    # upload the embeddings
    print("Starting upload of embeddings.")
    api_workflow_client.upload_embeddings(
        path_to_embeddings_csv=path_to_embeddings_csv, name="embedding_1"
    )
    print("Finished upload of embeddings.")

    return api_workflow_client


def t_est_api_with_matrix(
    path_to_dataset: str, token: str, dataset_name: str = "test_api_from_pip"
):
    no_samples = len(LightlyDataset(input_dir=path_to_dataset).dataset.samples)
    assert no_samples >= 100, "Test needs at least 100 samples in the dataset!"

    api_workflow_client = create_new_dataset_with_embeddings(
        path_to_dataset=path_to_dataset, token=token, dataset_name=dataset_name
    )

    api_workflow_client.delete_dataset_by_id(api_workflow_client.dataset_id)

    print(
        "Success of the complete test suite! The dataset on the server was deleted again."
    )


if __name__ == "__main__":
    if len(sys.argv) == 1 + 2:
        path_to_dataset, token = (sys.argv[1 + i] for i in range(2))
    else:
        raise ValueError(
            "ERROR in number of command line arguments, must be 2."
            "Example: python test_api path/to/dataset LIGHTLY_TOKEN"
        )

    t_est_api_with_matrix(path_to_dataset=path_to_dataset, token=token)
