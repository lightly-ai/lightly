import cv2
import itertools
from PIL import Image
import requests
import torch
import numpy as np

from lightly.data._helpers import IMG_EXTENSIONS
from lightly.data._helpers import VIDEO_EXTENSIONS

def _filter_extensions(samples, extensions):
    return [s for s in samples if s[0].endswith(extensions)]


class IterableDataset(torch.utils.data.IterableDataset):
    
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def __iter__(self):
        samples = self.samples

        # check if we have multiple workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # split work evenly among workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            samples = samples[worker_id::num_workers]
        
        return iter(self._load_samples(samples))

    def _load_samples(self, samples):
        """
        Loads and returns data for every sample
        """
        raise NotImplementedError


class ImageIterableDataset(IterableDataset):

    def __init__(self, samples):
        """
        Args:
            samples: list(tuple)
                Each tuple should contain (filename, url)
        """
        super().__init__(samples)
        self.samples = _filter_extensions(self.samples, IMG_EXTENSIONS)

    def _load_samples(self, samples):
        session = requests.Session()
        for filename, url in samples:
            response = session.get(url, stream=True)
            image = Image.open(response.raw)
            image = np.array(image)
            yield image, filename, 0


class VideoIterableDataset(IterableDataset):

    def __init__(self, samples):
        super().__init__(samples)
        self.samples = _filter_extensions(self.samples, VIDEO_EXTENSIONS)

    def _load_samples(self, samples):
        # each video is an iterator over its frames
        videos = map(self._load_video, samples)
        # flatten videos so we have a single iterator
        # over all frames of all videos
        return itertools.chain.from_iterable(videos)

    def _load_video(self, file_info):
        """Generates (frame, filename, frame_idx) tuples for a video"""
        filename = file_info['filename']
        url = file_info['url']
        video = cv2.VideoCapture(url)
        frame_exists = True
        frame_idx = 0
        while frame_exists:
            frame_exists, frame = video.read()
            frame = np.array(frame)
            yield frame, filename, frame_idx
            frame_idx += 1
