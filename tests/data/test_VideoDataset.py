import unittest
import os
import shutil
import numpy as np
import tempfile
import warnings
import PIL
import torchvision

from lightly.data._video import VideoDataset, _make_dataset
import cv2

class TestVideoDataset(unittest.TestCase):

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
        for backend in ['pyav', 'video_reader']:
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
        for backend in ['pyav', 'video_reader']:
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

