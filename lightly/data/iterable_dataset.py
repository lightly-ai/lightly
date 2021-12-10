import itertools
import requests
import torch

from lightly.api import download
from lightly.data import _helpers

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
        """Loads and returns data for every sample"""
        raise NotImplementedError

    def _target(self, filename):
        """Get weak label based on filename"""
        raise NotImplementedError


class ImageIterableDataset(IterableDataset):

    def __init__(self, samples):
        """
        Args:
            samples: list(tuple)
                Each tuple should contain (filename, url)
        """
        super().__init__(samples)
        self.samples = _filter_extensions(self.samples, _helpers.IMG_EXTENSIONS)

    def _load_samples(self, samples):
        session = requests.Session()
        for filename, url in samples:
            image = download.download_image(url, session)
            target = self._target(filename)
            yield image, filename, target

    def __getitem__(self, index, session=None):
        filename, url = self.samples[index]
        image = download.download_image(url, session)
        target = self._target(filename)
        return image, filename, target

    def _target(self, filename):
        # TODO
        return 0


class VideoIterableDataset(IterableDataset):

    def __init__(self, samples):
        super().__init__(samples)
        self.samples = _filter_extensions(
            self.samples, _helpers.VIDEO_EXTENSIONS
        )

    def _load_samples(self, samples):
        # each video is an iterator over its frames
        videos = map(self._load_video, samples)
        # flatten videos so we have a single iterator
        # over all frames of all videos
        return itertools.chain.from_iterable(videos)

    def _load_video(self, file_info):
        """Generates (frame, filename, frame_idx) tuples for a video"""
        filename, url = file_info
        frames = download.download_all_video_frames(url)
        for frame_index, frame in enumerate(frames):
            yield frame, filename, frame_index

    def _target(self, filename):
        # TODO
        return 0
