import unittest

import numpy as np

from lightly.active_learning.scorers import classification
from lightly.active_learning.agents import agent
from lightly.active_learning.config import sampler_config


class TestAgent(unittest.TestCase):

    def test_Agent(self):

        agent_ = agent.ActiveLearningAgent(token='abc',dataset_id='def',path_to_embeddings='blub')

        no_samples = 500
        no_classes = 10
        no_labelled_samples = 10
        predictions = np.random.random(size=(no_samples, no_classes))
        predictions_normalized = predictions / np.sum(predictions, axis=1)[:,np.newaxis]
        scorer = classification.ScorerClassification(model_output=predictions_normalized)

        labelled_ids = list(range(no_labelled_samples))

        sampler_name = 'random'
        batch_size = 64
        sampler_config_ = sampler_config.SamplerConfig(name=sampler_name, batch_size=batch_size)

        chosen_samples = agent_.sample(sampler_config=sampler_config_, al_scorer=scorer, labelled_ids=labelled_ids)

        assert len(chosen_samples) == batch_size
