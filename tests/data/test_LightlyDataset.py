import unittest
import os
import shutil
import torchvision
import tempfile
from lightly.data import LightlyDataset


class TestLightlyDataset(unittest.TestCase):

    def ensure_dir(self, path_to_folder: str):
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)

    def setUp(self):
        self.available_dataset_names = ['cifar10',
                                        #'cifar100',
                                        #'cityscapes',
                                        #'stl10',
                                        #'voc07-seg',
                                        #'voc12-seg',
                                        #'voc07-det',
                                        #'voc12-det]
                                        ]

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

    def test_create_lightly_dataset_from_folder(self):
        n_subfolders = 5
        n_samples_per_subfolder = 10
        n_tot_files = n_subfolders * n_samples_per_subfolder

        dataset_dir, folder_names, sample_names = self.create_dataset(
            n_subfolders,
            n_samples_per_subfolder
        )

        dataset = LightlyDataset(from_folder=dataset_dir)
        filenames = dataset.get_filenames()

        fnames = []
        for dir_name in folder_names:
            for fname in sample_names:
                fnames.append(os.path.join(dir_name, fname))

        self.assertEqual(len(filenames), n_tot_files)
        self.assertEqual(len(dataset), n_tot_files)
        self.assertListEqual(sorted(fnames), sorted(filenames))

        shutil.rmtree(dataset_dir)

    def test_create_lightly_dataset_from_folder_nosubdir(self):

        # create a dataset
        n_tot = 100
        dataset = torchvision.datasets.FakeData(size=n_tot,
                                                image_size=(3, 32, 32))

        tmp_dir = tempfile.mkdtemp()
        sample_names = [f'img_{i}.jpg' for i in range(n_tot)]
        for sample_idx in range(n_tot):

            data = dataset[sample_idx]
            path = os.path.join(tmp_dir, sample_names[sample_idx])
            data[0].save(path)

        # create lightly dataset
        dataset = LightlyDataset(from_folder=tmp_dir)
        filenames = dataset.get_filenames()

        # tests
        self.assertEqual(len(filenames), n_tot)
        self.assertEqual(len(dataset), n_tot)
        self.assertListEqual(sorted(sample_names), sorted(filenames))

        for i in range(n_tot):
            sample, target, fname = dataset[i]

    def test_create_lightly_dataset_from_torchvision(self):
        tmp_dir = tempfile.mkdtemp()

        for dataset_name in self.available_dataset_names:
            dataset = LightlyDataset(root=tmp_dir, name=dataset_name)
            self.assertIsNotNone(dataset)

    def test_not_existing_torchvision_dataset(self):
        list_of_non_existing_names = [
            'a-random-dataset',
            'cifar-100',
            'googleset_ 200'
        ]
        tmp_dir = tempfile.mkdtemp() 
        for dataset_name in list_of_non_existing_names:
            with self.assertRaises(ValueError):
                LightlyDataset(root=tmp_dir, name=dataset_name)

    def test_not_existing_folder_dataset(self):
        with self.assertRaises(ValueError):
            LightlyDataset(
                from_folder='/a-random-hopefully-non/existing-path-to-nowhere/'
            )
