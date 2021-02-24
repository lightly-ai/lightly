import csv
import os
from typing import *

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from lightly.active_learning.agents.agent import ActiveLearningAgent
from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.classification import ScorerClassification
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import SamplingMethod


class CSVEmbeddingDataset:
    def __init__(self, path_to_embeddings_csv: str):
        with open(path_to_embeddings_csv, 'r') as f:
            data = csv.reader(f)

            rows = list(data)
            header_row = rows[0]
            rows_without_header = rows[1:]

            index_filenames = header_row.index('filenames')
            filenames = [row[index_filenames] for row in rows_without_header]

            index_labels = header_row.index('labels')
            labels = [row[index_labels] for row in rows_without_header]

            embeddings = rows_without_header
            indexes_to_delete = sorted([index_filenames, index_labels], reverse=True)
            for embedding_row in embeddings:
                for index_to_delete in indexes_to_delete:
                    del embedding_row[index_to_delete]

        self.dataset = dict([(filename, (np.array(embedding_row, dtype=float), int(label)))
                             for filename, embedding_row, label in zip(filenames, embeddings, labels)])

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


def real_active_learning_run(path_to_dataset: str, token: str, dataset_id: str,
                             path_to_train_embeddings_csv: str, path_to_test_embeddings_csv: str,
                             method: SamplingMethod = SamplingMethod.CORAL,
                             ratios: List[str] = [0.01, 0.03, 0.1]):
    # define the api_client and api_workflow
    api_workflow_client = ApiWorkflowClient(host="https://api-dev.lightly.ai", token=token, dataset_id=dataset_id)

    # 1. upload the images to the dataset and create the initial tag
    no_tags_on_server = len(api_workflow_client.tags_api.get_tags_by_dataset_id(dataset_id=dataset_id))
    if no_tags_on_server == 0:
        api_workflow_client.upload_dataset(input=path_to_dataset)
    else:
        print("Skip upload of dataset: already uploaded.")

    # 2. upload the embeddings of the dataset
    api_workflow_client.upload_embeddings(path_to_embeddings_csv=path_to_train_embeddings_csv, name=f"embedding_1")

    # define the active learning agent and dataset
    agent = ActiveLearningAgent(api_workflow_client)
    training_set = CSVEmbeddingDataset(path_to_embeddings_csv=path_to_train_embeddings_csv)
    test_set = CSVEmbeddingDataset(path_to_embeddings_csv=path_to_test_embeddings_csv)
    no_samples_total = len(training_set.dataset.items())

    al_scorer = None

    for iteration, ratio in enumerate(ratios):
        print(f"Beginning with iteration {iteration}")
        # 3. Perform a sampling
        n_samples = int(ratio * no_samples_total)
        method_here = SamplingMethod.CORESET if iteration == 0 and method == SamplingMethod.CORAL else method
        sampler_config = SamplerConfig(method=method_here, n_samples=n_samples)
        if al_scorer is None:
            agent.query(sampler_config=sampler_config)
        else:
            agent.query(sampler_config=sampler_config,al_scorer=al_scorer)

        # 4. get the features and labels of the labeled_set
        features = training_set.get_features(agent.labeled_set)
        labels = training_set.get_labels(agent.labeled_set)

        # 5.1 train a classifier
        classifier = KNeighborsClassifier(n_neighbors=3)
        classifier.fit(features, labels)
        # 5.2 compute the new accuracy
        test_features, test_labels = test_set.all_features_labels()
        accuracy = classifier.score(test_features, test_labels)
        print(f"Accuracy of classifier: {accuracy:.3}")

        if iteration != len(ratios)-1:  # skip the following for the last iteration
            # 5.3 predict on the unlabeled set
            predictions = classifier.predict_proba(training_set.get_features(agent.unlabeled_set))

            # 6. Save the predictions in a scorer
            al_scorer = ScorerClassification(model_output=predictions)


if __name__ == "__main__":
    path_to_dataset = "/Users/malteebnerlightly/Documents/datasets/clothing-dataset-small-master/train"
    token = os.getenv("TOKEN")
    dataset_id = "603606ed2176760032373065"
    path_to_train_embeddings_csv = "/Users/malteebnerlightly/Documents/datasets/clothing-dataset-small-master/train/lightly_outputs/2021-02-23/23-38-25/embeddings.csv"
    path_to_test_embeddings_csv = "/Users/malteebnerlightly/Documents/datasets/clothing-dataset-small-master/test/lightly_outputs/2021-02-23/23-41-09/embeddings.csv"
    for i in range(1):
        print(f"ITERATION {i}:")
        real_active_learning_run(path_to_dataset, token, dataset_id,
                                 path_to_train_embeddings_csv=path_to_train_embeddings_csv,
                                 path_to_test_embeddings_csv=path_to_test_embeddings_csv)
        print("")