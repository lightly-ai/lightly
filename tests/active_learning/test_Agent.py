import tempfile
import unittest
import os

import lightly
import numpy as np
import torchvision

from lightly.active_learning.scorers import classification
from lightly.active_learning.agents import agent
from lightly.active_learning.config import sampler_config


class TestAgent(unittest.TestCase):

    def setup(self, n_data=1000):

        folder_path = tempfile.mkdtemp()
        self.path_to_embeddings = os.path.join(
            folder_path,
            'embeddings.csv'
        )

        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        labels = [0] * len(sample_names)

        lightly.utils.save_embeddings(
            self.path_to_embeddings,
            np.random.randn(n_data, 16),
            labels,
            sample_names
        )

    def test_Agent(self):
        self.setup()

        token = os.environ.get('TEST_TOKEN')
        dataset_id = os.environ.get('TEST_DATASET_ID')
        agent_ = agent.ActiveLearningAgent(token=token, dataset_id=dataset_id,
                                           path_to_embeddings=self.path_to_embeddings)

        no_samples = 500
        no_classes = 10
        no_labelled_samples = 10
        predictions = np.random.random(size=(no_samples, no_classes))
        predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
        scorer = classification.ScorerClassification(model_output=predictions_normalized)

        labelled_ids = list(range(no_labelled_samples))

        sampler_name = 'random'
        batch_size = 64
        sampler_config_ = sampler_config.SamplerConfig(name=sampler_name, batch_size=batch_size)

        chosen_samples = agent_.sample(sampler_config=sampler_config_, al_scorer=scorer, labelled_ids=labelled_ids)

        assert len(chosen_samples) == batch_size
