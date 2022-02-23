import unittest
import os
import re
import shutil

import numpy as np
import torchvision
import tempfile
import pytest

from lightly.core import train_model_and_embed_images


class TestCore(unittest.TestCase):

    def ensure_dir(self, path_to_folder: str):
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)

    def create_dataset(self, n_subfolders=5, n_samples_per_subfolder=20):
        n_tot = n_subfolders * n_samples_per_subfolder
        dataset = torchvision.datasets.FakeData(size=n_tot,
                                                image_size=(3, 32, 32))

        tmp_dir = tempfile.mkdtemp()

        folder_names = [f'folder_{i}' for i in range(n_subfolders)]
        sample_names = [f'img_{i}.jpg' for i in range(n_samples_per_subfolder)]

        for folder_idx in range(n_subfolders):
            for sample_idx in range(n_samples_per_subfolder):
                idx = (folder_idx * n_subfolders) + sample_idx
                data = dataset[idx]

                self.ensure_dir(os.path.join(tmp_dir,
                                             folder_names[folder_idx]))

                data[0].save(os.path.join(tmp_dir,
                                          folder_names[folder_idx],
                                          sample_names[sample_idx]))
        self.dataset_dir = tmp_dir
        return tmp_dir, folder_names, sample_names


    #@pytest.mark.slow
    def test_train_and_embed(self):
        n_subfolders = 3
        n_samples_per_subfolder = 3
        n_samples = n_subfolders * n_samples_per_subfolder

        # embed, no overwrites
        dataset_dir, _, _ = self.create_dataset(
            n_subfolders,
            n_samples_per_subfolder
        )

        # train, one overwrite
        embeddings, labels, filenames = train_model_and_embed_images(
            input_dir=dataset_dir,
            trainer={'max_epochs': 1},
            loader={'num_workers': 0},
        )
        self.assertEqual(len(embeddings), n_samples)
        self.assertEqual(len(labels), n_samples)
        self.assertEqual(len(filenames), n_samples)
        self.assertIsInstance(embeddings[0], np.ndarray)
        self.assertIsInstance(int(labels[0]), int)  # see if casting to int works
        self.assertIsInstance(filenames[0], str)


    def tearDown(self) -> None:
        shutil.rmtree(self.dataset_dir)
        pattern = '(.*)?.ckpt$'
        for root, dirs, files in os.walk(os.getcwd()):
            for file in filter(lambda x: re.match(pattern, x), files):
                os.remove(os.path.join(root, file))
