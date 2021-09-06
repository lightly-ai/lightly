import unittest
import os
import random
import shutil
import torchvision
import tempfile
import warnings
import numpy as np
from lightly.data import LightlyDataset

from lightly.data._utils import check_images
from lightly.utils.io import INVALID_FILENAME_CHARACTERS

try:
    from lightly.data._video import VideoDataset
    import av
    import cv2

    VIDEO_DATASET_AVAILABLE = True
except Exception:
    VIDEO_DATASET_AVAILABLE = False


class TestLightlyDataset(unittest.TestCase):

    def ensure_dir(self, path_to_folder: str):
        os.makedirs(path_to_folder, exist_ok=True)

    def setUp(self):
        self.available_dataset_names = ['cifar10',
                                        # 'cifar100',
                                        # 'cityscapes',
                                        # 'stl10',
                                        # 'voc07-seg',
                                        # 'voc12-seg',
                                        # 'voc07-det',
                                        # 'voc12-det]
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

    def create_video_dataset(self, n_videos=5, n_frames_per_video=10, w=32, h=32, c=3):

        self.n_videos = n_videos
        self.n_frames_per_video = n_frames_per_video

        self.input_dir = tempfile.mkdtemp()
        self.ensure_dir(self.input_dir)
        self.frames = (np.random.randn(n_frames_per_video, w, h, c) * 255).astype(np.uint8)
        self.extensions = ('.avi')

        for i in range(5):
            path = os.path.join(self.input_dir, f'output-{i}.avi')
            print(path)
            out = cv2.VideoWriter(path, 0, 1, (w, h))
            for frame in self.frames:
                out.write(frame)
            out.release()

    def test_create_lightly_dataset_from_folder(self):
        n_subfolders = 5
        n_samples_per_subfolder = 10
        n_tot_files = n_subfolders * n_samples_per_subfolder

        dataset_dir, folder_names, sample_names = self.create_dataset(
            n_subfolders,
            n_samples_per_subfolder
        )

        dataset = LightlyDataset(input_dir=dataset_dir)
        filenames = dataset.get_filenames()

        fnames = []
        for dir_name in folder_names:
            for fname in sample_names:
                fnames.append(os.path.join(dir_name, fname))

        self.assertEqual(len(filenames), n_tot_files)
        self.assertEqual(len(dataset), n_tot_files)
        self.assertListEqual(sorted(fnames), sorted(filenames))

        out_dir = tempfile.mkdtemp()
        dataset.dump(out_dir)
        self.assertEqual(
            sum(len(os.listdir(os.path.join(out_dir, subdir))) for subdir in os.listdir(out_dir)),
            len(dataset),
        )

        shutil.rmtree(dataset_dir)
        shutil.rmtree(out_dir)

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
        dataset = LightlyDataset(input_dir=tmp_dir)
        filenames = dataset.get_filenames()

        # tests
        self.assertEqual(len(filenames), n_tot)
        self.assertEqual(len(dataset), n_tot)
        self.assertListEqual(sorted(sample_names), sorted(filenames))

        for i in range(n_tot):
            sample, target, fname = dataset[i]

    def test_create_lightly_dataset_with_invalid_char_in_filename(self):

        # create a dataset
        n_tot = 100
        dataset = torchvision.datasets.FakeData(size=n_tot,
                                                image_size=(3, 32, 32))

        for invalid_char in INVALID_FILENAME_CHARACTERS:
            with self.subTest(msg=f"invalid_char: {invalid_char}"):
                tmp_dir = tempfile.mkdtemp()
                sample_names = [f'img_,_{i}.jpg' for i in range(n_tot)]
                for sample_idx in range(n_tot):
                    data = dataset[sample_idx]
                    path = os.path.join(tmp_dir, sample_names[sample_idx])
                    data[0].save(path)

                # create lightly dataset
                    with self.assertRaises(ValueError):
                        dataset = LightlyDataset(input_dir=tmp_dir)

    def test_check_images(self):

        # create a dataset
        tmp_dir = tempfile.mkdtemp()
        n_healthy = 100
        n_corrupt = 20

        dataset = torchvision.datasets.FakeData(size=n_healthy,
                                                image_size=(3, 32, 32))
        sample_names = [f'img_{i}.jpg' for i in range(n_healthy)]
        for sample_name, data in zip(sample_names, dataset):
            path = os.path.join(tmp_dir, sample_name)
            data[0].save(path)

        corrupt_sample_names = [f'img_{i}.jpg' for i in range(n_healthy, n_healthy + n_corrupt)]
        for sample_name in corrupt_sample_names:
            path = os.path.join(tmp_dir, sample_name)
            with open(path, 'a') as f:
                f.write('this_is_not_an_image')

        # tests
        healthy_images, corrupt_images = check_images(tmp_dir)
        assert (len(healthy_images) == n_healthy)
        assert (len(corrupt_images) == n_corrupt)

    def test_not_existing_folder_dataset(self):
        with self.assertRaises(ValueError):
            LightlyDataset(
                '/a-random-hopefully-non/existing-path-to-nowhere/'
            )

    def test_from_torch_dataset(self):
        _dataset = torchvision.datasets.FakeData(size=1, image_size=(3, 32, 32))
        dataset = LightlyDataset.from_torch_dataset(_dataset)
        self.assertEqual(len(_dataset), len(dataset))
        self.assertEqual(len(dataset.get_filenames()), len(dataset))



    def test_video_dataset(self):

        if not VIDEO_DATASET_AVAILABLE:
            tmp_dir = tempfile.mkdtemp()
            # simulate a video
            # the video dataset will check to see whether there exists a file
            # with a video extension, it's enough to fake a video file here
            path = os.path.join(tmp_dir, 'my_file.png')
            dataset = torchvision.datasets.FakeData(size=1, image_size=(3, 32, 32))
            image, _ = dataset[0]
            image.save(path)
            os.rename(path, os.path.join(tmp_dir, 'my_file.avi'))
            with self.assertRaises(ImportError):
                dataset = LightlyDataset(input_dir=tmp_dir)

            warnings.warn(
                'Did not test video dataset because of missing requirements')
            shutil.rmtree(tmp_dir)
            return

        self.create_video_dataset()
        dataset = LightlyDataset(input_dir=self.input_dir)

        out_dir = tempfile.mkdtemp()
        dataset.dump(out_dir)
        self.assertEqual(len(os.listdir(out_dir)), len(dataset))

    def test_transform_setter(self, dataset: LightlyDataset = None):

        if dataset is None:
            tmp_dir, _, _ = self.create_dataset()
            dataset = LightlyDataset(input_dir=tmp_dir)
        # the transform of both datasets should be None
        self.assertIsNone(dataset.transform)
        self.assertIsNone(dataset.dataset.transform)
        # use the setter
        dataset.transform = torchvision.transforms.ToTensor()
        # assert that the transform is set in the nested dataset
        self.assertIsNotNone(dataset.transform)
        self.assertIsNotNone(dataset.dataset.transform)
