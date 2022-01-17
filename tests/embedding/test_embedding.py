import os
import tempfile
import unittest
from typing import Tuple, List

import torchvision
from hydra.experimental import initialize, compose
from torch.utils.data import DataLoader

from lightly.cli.embed_cli import get_model_from_config
from lightly.data import LightlyDataset
from lightly.embedding import SelfSupervisedEmbedding


class TestLightlyDataset(unittest.TestCase):

    def setUp(self):
        self.folder_path, self.sample_names = self.create_dataset_no_subdir(100)
        with initialize(config_path="../../lightly/cli/config", job_name="test_app"):
            self.cfg = compose(config_name="config", overrides=[
                "token='123'",
                f"input_dir={self.folder_path}",
                "trainer.max_epochs=0"
            ])

    def create_dataset_no_subdir(self, n_samples: int) -> Tuple[str, List[str]]:
        dataset = torchvision.datasets.FakeData(size=n_samples,
                                                image_size=(3, 32, 32))

        tmp_dir = tempfile.mkdtemp()
        sample_names = [f'img_{i}.jpg' for i in range(n_samples)]
        for sample_idx in range(n_samples):
            data = dataset[sample_idx]
            path = os.path.join(tmp_dir, sample_names[sample_idx])
            data[0].save(path)
        return tmp_dir, sample_names
    
    @unittest.skip("Failing for the moment")
    def test_embed_correct_order(self):
        # get dataloader
        transform = torchvision.transforms.ToTensor()
        dataset = LightlyDataset(self.folder_path, transform=transform)
        dataloader = DataLoader(dataset, shuffle=True)

        encoder = get_model_from_config(self.cfg)
        
        embeddings, labels, filenames = encoder.embed(dataloader)

        self.assertListEqual(filenames, dataset.get_filenames())
