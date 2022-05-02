from fractions import Fraction
import unittest
import os
import shutil
from unittest import mock
import numpy as np
import tempfile
import warnings
import PIL
import torch
import torchvision

from lightly.data import LightlyDataset, NonIncreasingTimestampError
from lightly.data._video import (
    VideoDataset, 
    _make_dataset,
    _find_non_increasing_timestamps,
)

import cv2

try:
    import av
    PYAV_AVAILABLE = True
except ModuleNotFoundError:
    PYAV_AVAILABLE = False

VIDEO_BACKENDS = ['pyav', 'video_reader']

class TestVideoDataset(unittest.TestCase):

    def setUp(self):
        if not PYAV_AVAILABLE:
            self.skipTest('PyAV not available')

    def ensure_dir(self, path_to_folder: str):
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)

    def create_dataset(self, n_videos=5, n_frames_per_video=10, w=32, h=32, c=3):

        self.n_videos = n_videos
        self.n_frames_per_video = n_frames_per_video
    
        self.input_dir = tempfile.mkdtemp()
        self.ensure_dir(self.input_dir)
        self.frames = (np.random.randn(n_frames_per_video, w, h, c) * 255).astype(np.uint8)
        self.extensions = ('.avi')

        for i in range(n_videos):
            path = os.path.join(self.input_dir, f'output-{i}.avi')
            print(path)
            out = cv2.VideoWriter(path, 0, 1, (w, h))
            for frame in self.frames:
                out.write(frame)
            out.release()

    def test_video_similar_timestamps_for_different_backends(self):

        self.create_dataset()

        timestamps = []
        offsets = []
        backends = []

        # iterate through different backends
        for backend in VIDEO_BACKENDS:
            torchvision.set_video_backend(backend)

            _, video_timestamps, video_offsets, _ = \
                _make_dataset(self.input_dir, extensions=self.extensions)
            timestamps.append(video_timestamps)
            offsets.append(video_offsets)
            backends.append(backend)
        
        # make sure backends don't match (sanity check)
        self.assertNotEqual(backends[0], backends[1])

        # we expect the same timestamps and offsets
        self.assertEqual(timestamps[0], timestamps[1])
        self.assertEqual(offsets[0], offsets[1])

        shutil.rmtree(self.input_dir)


    def test_video_dataset_from_folder(self):


        self.create_dataset()

        # iterate through different backends
        for backend in VIDEO_BACKENDS:
            torchvision.set_video_backend(backend)

            # create dataset
            dataset = VideoDataset(self.input_dir, extensions=self.extensions)
            
            # __len__
            self.assertEqual(len(dataset), self.n_frames_per_video * self.n_videos)

            # __getitem__
            for i in range(len(dataset)):
                frame, label = dataset[i]
                self.assertIsInstance(frame, PIL.Image.Image)
                self.assertEqual(label, i // self.n_frames_per_video)

            # get_filename
            for i in range(len(dataset)):
                frame, label = dataset[i]
                filename = dataset.get_filename(i)
                print(filename)
                self.assertTrue(
                    filename.endswith(
                        f"-{(i % self.n_frames_per_video):02d}-avi.png"
                    )
                )
        
        shutil.rmtree(self.input_dir)

    def test_video_dataset_no_read_rights(self):
        self.create_dataset()

        with self.subTest("no read rights files"):
            for subdir, dirs, files in os.walk(self.input_dir):
                for filename in files:
                    filepath = os.path.join(self.input_dir, filename)
                    os.chmod(filepath, 0o000)
            with self.assertRaises(PermissionError):
                dataset = LightlyDataset(self.input_dir)

        with self.subTest("no read rights subdirs"):
            for subdir, dirs, files in os.walk(self.input_dir):
                os.chmod(subdir, 0o000)
            with self.assertRaises(PermissionError):
                dataset = LightlyDataset(self.input_dir)

        with self.subTest("no read rights root"):
            os.chmod(self.input_dir, 0o000)
            with self.assertRaises(PermissionError):
                dataset = LightlyDataset(self.input_dir)

    def test_video_dataset_non_increasing_timestamps(self):
        self.create_dataset(n_videos=2, n_frames_per_video=5)
        
        # overwrite the _make_dataset function to return a wrong timestamp
        def _make_dataset_with_non_increasing_timestamps(*args):
            video_instances, timestamps, offsets, fpss = _make_dataset(*args)
            # set timestamp of 4th frame in 1st video to timestamp of 2nd frame.
            timestamps[0][3] = timestamps[0][1]
            return video_instances, timestamps, offsets, fpss

        with mock.patch('lightly.data._video._make_dataset', _make_dataset_with_non_increasing_timestamps):
            for backend in VIDEO_BACKENDS:
                torchvision.set_video_backend(backend)

                # getting frame at wrong timestamp should throw an exception
                dataset = VideoDataset(self.input_dir, extensions=self.extensions)
                for i in range(len(dataset)):
                    if i == 3:
                        # frame with wrong timestamp
                        with self.assertRaises(NonIncreasingTimestampError):
                            dataset[i]
                    else:
                        dataset[i]

                # Getting frame at wrong timestamp should throw an exception
                # from dataloader but not break the dataloader itself. Future
                # calls to next() should still work.
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    num_workers=2,
                    batch_size=None,
                    collate_fn=lambda x: x
                )
                dataloader_iter = iter(dataloader)
                for i in range(len(dataset)):
                    if i == 3:
                        # frame with wrong timestamp
                        with self.assertRaises(NonIncreasingTimestampError):
                            next(dataloader_iter)
                    else:
                        next(dataloader_iter)

                # disable exception, should be able to load all frames
                dataset.exception_on_non_increasing_timestamp = False
                total_frames = 0
                for _ in dataset:
                    total_frames += 1
                self.assertEqual(total_frames, len(dataset))


    def test_video_dataset_dataloader(self):
        self.create_dataset()
        for backend in VIDEO_BACKENDS:
            torchvision.set_video_backend(backend)
            dataset = VideoDataset(self.input_dir, extensions=self.extensions)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                num_workers=2,
                batch_size=3,
                shuffle=True,
                collate_fn=lambda x: x,
            )
            for batch in dataloader:
                pass
    
    
    def test_find_non_increasing_timestamps(self):
        # no timestamps
        non_increasing = _find_non_increasing_timestamps([])
        self.assertListEqual(non_increasing, [])

        # single timestamp
        timestamps = [Fraction(0, 1)]
        expected = [False]
        non_increasing = _find_non_increasing_timestamps(timestamps)
        self.assertListEqual(non_increasing, expected)

        # all timestamps increasing
        timestamps = [Fraction(0, 1), Fraction(1, 1), Fraction(2, 1)]
        expected = [False, False, False]
        non_increasing = _find_non_increasing_timestamps(timestamps)
        self.assertListEqual(non_increasing, expected)

        # all timestamps equal
        timestamps = [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)]
        expected = [False, True, True]
        non_increasing = _find_non_increasing_timestamps(timestamps)
        self.assertListEqual(non_increasing, expected)
        
        # some timestamps equal and some decreasing
        timestamps = [Fraction(-1, 1), Fraction(0, 1), Fraction(1, 1), Fraction(2, 3), Fraction(2, 3), Fraction(2, 1), Fraction(3, 2)]
        expected = [False, False, False, True, True, False, True]
        non_increasing = _find_non_increasing_timestamps(timestamps)
        self.assertListEqual(non_increasing, expected)
