""" """

#
# 

import os
from PIL import Image
from torchvision import datasets

from torchvision.io import read_video, read_video_timestamps


def _video_loader(path, timestamp, pts_unit='sec'):
    """

    """
    # random access read from video
    frame, _, _ = read_video(path,
                             start_pts=timestamp,
                             end_pts=timestamp,
                             pts_unit=pts_unit)
    # read_video returns tensor of shape 1 x W x H x C
    frame = frame.squeeze()
    # convert to PIL image
    # TODO: can it be on CUDA? -> need to move it to CPU first
    image = Image.fromarray(frame.numpy())

    return image


def _make_dataset(directory, extensions=None, is_valid_file=None, pts_unit='sec'):
    """

    """

    if extensions is not None:
        def _is_valid_file(filename):
            return filename.lower().endswith(extensions)

    # find all instances
    instances = []
    for fname in os.listdir(directory):

        if not _is_valid_file(fname):
            continue

        path = os.path.join(directory, fname)
        instances.append(path)

    # get timestamps
    timestamps, fpss = [], []
    for instance in instances:
        ts, fps = read_video_timestamps(instance, pts_unit=pts_unit)
        timestamps.append(ts)
        fpss.append(fps)

    # get offsets
    offsets = [len(ts) for ts in timestamps]
    offsets = [0] + offsets[:-1]
    for i in range(1, len(offsets)):
        offsets[i] = offsets[i-1] + offsets[i] # cumsum

    return instances, timestamps, offsets, fpss


class VideoDataset(datasets.VisionDataset):
    """

    """

    def __init__(self, root, loader=_video_loader, extensions=None,
                 transform=None, target_transform=None, is_valid_file=None):
        
        super(VideoDataset, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        videos, video_timestamps, offsets, fpss = _make_dataset(
            self.root, extensions, is_valid_file)

        self.extensions = extensions
        self.loader = loader

        self.videos = videos
        self.video_timestamps = video_timestamps
        self.offsets = offsets
        self.fpss = fpss

    def __getitem__(self, index):
        """

        """
        if index < 0 or index >= self.__len__():
            raise IndexError(f'Index {index} is out of bounds for VideoDataset'
                             f' of size {self.__len__()}.')

        # find video of the frame
        i = 0
        while i < len(self.offsets) - 1:
            if self.offsets[i] >= index:
                break
            i = i + 1

        # find and return the frame as PIL image
        target = i
        sample = self.loader(self.videos[i],
                             self.video_timestamps[i][index - self.offsets[i]])
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        """

        """
        return sum((len(ts) for ts in self.video_timestamps))

    def _get_filename(self, index):
        """

        """
        if index < 0 or index >= self.__len__():
            raise IndexError(f'Index {index} is out of bounds for VideoDataset'
                             f' of size {self.__len__()}.')
    
        # find video of the frame
        i = 0
        while i < len(self.offsets) - 1:
            if self.offsets[i] >= index:
                break
            i = i + 1
        
        filename = '.'.join(self.videos[i].split('.')[:-1])
        timestamp = float(self.video_timestamps[i][index - self.offsets[i]])
        return '%s-%.8fs.png' % (filename, timestamp)
