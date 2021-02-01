import tempfile
import unittest
import os

import lightly
import numpy as np
import torchvision

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers import classification
from lightly.active_learning.agents import agent
from lightly.active_learning.config import sampler_config
from lightly.openapi_generated.swagger_client import SamplingMethod


class TestAgent(unittest.TestCase):

    def setup(self, n_data=2000):

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

    #@unittest.skip("Part is not mocked yet, but tries to access the real server.")
    def test_Agent(self):
        self.setup()

        token = os.environ.get('TEST_TOKEN', 'f9b60358d529bdd824e3c2df')
        dataset_id = os.environ.get('TEST_DATASET_ID', '5ff6fa9b6580b3000acca8a8')
        os.environ.setdefault('TEST_TOKEN', token)
        os.environ.setdefault('TEST_DATASET_ID', dataset_id)
        agent_ = agent.ActiveLearningAgent(token=token, dataset_id=dataset_id,
                                           path_to_embeddings=self.path_to_embeddings)

        no_samples = 500
        no_classes = 10
        no_labelled_samples = 10
        predictions = np.random.random(size=(no_samples, no_classes))
        predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
        scorer = classification.ScorerClassification(model_output=predictions_normalized)

        labelled_ids = list(range(no_labelled_samples))

        batch_size = 64
        sampler_config = SamplerConfig(method=SamplingMethod.RANDOM, batch_size=batch_size, name='test_al_agent')

        chosen_samples = agent_.sample(sampler_config=sampler_config, al_scorer=scorer, labelled_ids=labelled_ids)

        assert len(chosen_samples) == batch_size
