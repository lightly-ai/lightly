import unittest
import os
import shutil
import numpy as np
import tempfile
import warnings
import PIL

try:
    from lightly.data._video import VideoDataset
    import cv2
    VIDEO_DATASET_AVAILABLE = True
except Exception:
    VIDEO_DATASET_AVAILABLE = False

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

        for i in range(5):
            path = os.path.join(self.input_dir, f'output-{i}.avi')
            print(path)
            out = cv2.VideoWriter(path, 0, 1, (w, h))
            for frame in self.frames:
                out.write(frame)
            out.release()

    def test_video_dataset_from_folder(self):

        if not VIDEO_DATASET_AVAILABLE:
            warnings.warn(
                'Did not test video dataset because of missing requirements')
            return

        self.create_dataset()

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
            self.assertTrue(
                filename.endswith(
                    f"-{float(i % self.n_frames_per_video):.8f}s-avi.png"
                )
            )
        
        shutil.rmtree(self.input_dir)
