import os
import tempfile
import unittest
from typing import Tuple, List

import numpy as np
import torchvision
from hydra.experimental import initialize, compose
from torch import manual_seed
from torch.utils.data import DataLoader

from lightly.cli._helpers import get_model_from_config
from lightly.data import LightlyDataset
from lightly.embedding import SelfSupervisedEmbedding


class TestLightlyDataset(unittest.TestCase):
    def setUp(self):
        self.folder_path, self.sample_names = self.create_dataset_no_subdir(10)
        with initialize(config_path='../../lightly/cli/config', job_name='test_app'):
            self.cfg = compose(
                config_name='config',
                overrides=[
                    'token="123"',
                    f'input_dir={self.folder_path}',
                    'trainer.max_epochs=0',
                ],
            )

    def create_dataset_no_subdir(self, n_samples: int) -> Tuple[str, List[str]]:
        dataset = torchvision.datasets.FakeData(size=n_samples, image_size=(3, 32, 32))

        tmp_dir = tempfile.mkdtemp()
        sample_names = [f'img_{i}.jpg' for i in range(n_samples)]
        for sample_idx in range(n_samples):
            data = dataset[sample_idx]
            path = os.path.join(tmp_dir, sample_names[sample_idx])
            data[0].save(path)
        return tmp_dir, sample_names

    def test_embed_correct_order(self):
        # get dataset and encoder
        transform = torchvision.transforms.ToTensor()
        dataset = LightlyDataset(self.folder_path, transform=transform)
        encoder = get_model_from_config(self.cfg)

        manual_seed(42)
        dataloader_1_worker = DataLoader(
            dataset, shuffle=True, num_workers=0, batch_size=4
        )
        embeddings_1_worker, labels_1_worker, filenames_1_worker = encoder.embed(
            dataloader_1_worker
        )

        manual_seed(43)
        dataloader_4_worker = DataLoader(
            dataset, shuffle=True, num_workers=4, batch_size=4
        )
        embeddings_4_worker, labels_4_worker, filenames_4_worker = encoder.embed(
            dataloader_4_worker
        )

        np.testing.assert_equal(embeddings_1_worker, embeddings_4_worker)
        np.testing.assert_equal(labels_1_worker, labels_4_worker)

        self.assertListEqual(filenames_1_worker, filenames_4_worker)
        self.assertListEqual(filenames_1_worker, dataset.get_filenames())
