from typing import List, Tuple
import itertools
import os

import requests
import torch
from lightly.api import download
from lightly.data import _helpers


def _filter_extensions(samples, extensions):
    return [s for s in samples if s[0].endswith(extensions)]


class LightlyIterableDataset(torch.utils.data.IterableDataset):
    
    def __init__(self, samples: List[Tuple[str, str]]):
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
        """Loads and returns data for every sample."""
        raise NotImplementedError

    def target(self, filename):
        """Get weak label based on filename."""
        raise NotImplementedError


class LightlyImageIterableDataset(LightlyIterableDataset):

    def __init__(self, samples):
        """
        Args:
            samples:
                List of tuple should contain (filename, url).
        """
        super().__init__(samples)
        self.samples = _filter_extensions(self.samples, _helpers.IMG_EXTENSIONS)
        classes, class_to_idx = self.find_classes(self.samples)
        self.classes = classes
        self.class_to_idx = class_to_idx

    def _load_samples(self, samples):
        session = requests.Session()
        for filename, url in samples:
            image = download.download_image(url, session)
            target = self.target(filename)
            yield image, filename, target

    def __getitem__(self, index, session=None):
        filename, url = self.samples[index]
        image = download.download_image(url, session)
        target = self.target(filename)
        return image, filename, target

    def find_classes(self, samples):
        filenames = [filename for filename, _ in samples]
        return find_classes_from_filenames(filenames)

    def target(self, filename):
        class_name = top_directory(filename)
        return self.class_to_idx[class_name]


class LightlyVideoIterableDataset(LightlyIterableDataset):

    def __init__(self, samples):
        super().__init__(samples)
        self.samples = _filter_extensions(
            self.samples, _helpers.VIDEO_EXTENSIONS
        )
        self.classes, self.class_to_idx = self.find_classes(self.samples)

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
            frame_name = self._frame_name(filename, frame_index)
            target = self.target(filename)
            yield frame, frame_name, target

    def _frame_name(self, filename, frame_index):
        video_name, video_format = os.path.splitext(filename)
        # remove '.' in '.mp4' or similar
        video_format = video_format[1:]
        return f'{video_name}-{frame_index:08}-{video_format}.png'

    def find_classes(self, samples):
        classes = sorted([filename for filename, _ in samples])
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def target(self, filename):
        return self.class_to_idx[filename]


def top_directory(filename):
    """Returns the top directory given a filename."""
    path = os.path.normpath(filename)
    parts = path.split(os.sep)
    first = parts[0]
    if len(parts) == 1:
        # file without a parent directory
        # for example 'file.png'
        return ""
    second = parts[1]
    if len(parts) == 2:
        if not first:
            # absolute path
            # for example '/file.png'
            return ""
        else:
            # single parent directory
            # for example 'top/file.png'
            return first

    if not first:
        # absolute path with parent
        # for example '/top/file.png'
        # or '/top/dir/file.png'
        return second
    else:
        # path with parent
        # for example 'top/dir/file.png'
        return first

def find_classes_from_filenames(filenames):
    """Finds classes based on the top level directory of the filenames"""
    filename_to_class = {filename: top_directory(filename) for filename in filenames}
    classes = sorted(list(set(filename_to_class.values())))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    return classes, class_to_idx
