import unittest
import os
import re
import shutil
import torchvision
import tempfile
from lightly import embed_images
from lightly import train_embedding_model
from lightly import train_model_and_embed_images


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
        return tmp_dir, folder_names, sample_names


    def test_train_and_embed(self):
        n_subfolders = 10
        n_samples_per_subfolder = 10

        # embed, no overwrites
        dataset_dir, _, _ = self.create_dataset(
            n_subfolders,
            n_samples_per_subfolder
        )

        # train, one overwrite
        trainer = {
            'max_epochs': 1
        }
        train_model_and_embed_images(
            input_dir=dataset_dir, trainer=trainer)
        shutil.rmtree(dataset_dir)
        pattern = 'lightly_epoch(.*)?.ckpt$'
        for root, dirs, files in os.walk(os.getcwd()):
            for file in filter(lambda x: re.match(pattern, x), files):
                os.remove(os.path.join(root, file))