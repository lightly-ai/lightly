""" Video Dataset """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
import av
from PIL import Image
from torchvision import datasets

from torchvision.io import read_video, read_video_timestamps


def _video_loader(path, timestamp, pts_unit='sec'):
    """Reads a frame from a video at a random timestamp.

    Args:
        path:
            Path to the video file.
        timestamp:
            The timestamp at which to retrieve the frame in seconds.
        pts_unit:
            Unit of the timestamp.

    """
    # random access read from video (slow)
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


def _make_dataset(directory,
                  extensions=None,
                  is_valid_file=None,
                  pts_unit='sec'):
    """Returns a list of all video files, timestamps, and offsets.

    Args:
        directory:
            Root directory path (should not contain subdirectories).
        extensions:
            Tuple of valid extensions.
        is_valid_file:
            Used to find valid files.
        pts_unit:
            Unit of the timestamps.

    Returns:
        A list of video files, timestamps, frame offsets, and fps.

    """

    # use filename to find valid files
    if extensions is not None:
        def _is_valid_file(filename):
            return filename.lower().endswith(extensions)

    # overwrite function to find valid files
    if is_valid_file is not None:
        _is_valid_file = is_valid_file

    # find all instances (no subdirectories)
    instances = []
    for fname in os.listdir(directory):

        # skip invalid files
        if not _is_valid_file(fname):
            continue

        # keep track of valid files
        path = os.path.join(directory, fname)
        instances.append(path)

    # get timestamps
    timestamps, fpss = [], []
    for instance in instances:
        ts, fps = read_video_timestamps(instance, pts_unit=pts_unit)
        timestamps.append(ts)
        fpss.append(fps)

    # get frame offsets
    offsets = [len(ts) for ts in timestamps]
    offsets = [0] + offsets[:-1]
    for i in range(1, len(offsets)):
        offsets[i] = offsets[i-1] + offsets[i] # cumsum

    return instances, timestamps, offsets, fpss


class VideoDataset(datasets.VisionDataset):
    """Implementation of a video dataset.

    The VideoDataset allows random reads from a video file without extracting
    all frames beforehand. This is more storage efficient but is slower.

    Attributes:
        root:
            Root directory path.
        loader:
            Function that loads file at path.
        extensions:
            Tuple of allowed extensions.
        transform:
            Function that takes a PIL image and returns transformed version
        target_transform:
            As transform but for targets
        is_valid_file:
            Used to check corrupt files

    """

    def __init__(self,
                 root,
                 loader=_video_loader,
                 extensions=None,
                 transform=None,
                 target_transform=None,
                 is_valid_file=None):
        
        super(VideoDataset, self).__init__(root,
                                           transform=transform,
                                           target_transform=target_transform)

        videos, video_timestamps, offsets, fpss = _make_dataset(
            self.root, extensions, is_valid_file)
        
        if len(videos) == 0:
            msg = 'Found 0 videos in folder: {}\n'.format(self.root)
            if extensions is not None:
                msg += 'Supported extensions are: {}'.format(
                    ','.join(extensions))
            raise RuntimeError(msg)

        self.extensions = extensions
        self.loader = loader

        self.videos = videos
        self.video_timestamps = video_timestamps
        self.offsets = offsets
        self.fpss = fpss

    def __getitem__(self, index):
        """Returns item at index.

        Finds the video of the frame at index with the help of the frame 
        offsets. Then, loads the frame from the video, applies the transforms,
        and returns the frame along with the index of the video (as target).

        Args:
            index:
                Index of the sample to retrieve.

        Returns:
            A tuple (sample, target) where target indicates the video index.

        Raises:
            IndexError if index is out of bounds.

        """
        if index < 0 or index >= self.__len__():
            raise IndexError(f'Index {index} is out of bounds for VideoDataset'
                             f' of size {self.__len__()}.')

        # find video of the frame
        i = len(self.offsets) - 1
        while (self.offsets[i] > index):
            i = i - 1

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
        """Returns the number of samples in the dataset.

        """
        return sum((len(ts) for ts in self.video_timestamps))

    def get_filename(self, index):
        """Returns a filename for the frame at index.

        The filename is created from the video filename, the timestamp, and
        the video format. E.g. when retrieving a sample from the video
        `my_video.mp4` at time 0.5s, the filename will be:

        >>> my_video-0.50000000s-mp4.png
    
        Args:
            index:
                Index of the frame to retrieve.

        Returns:
            The filename of the frame as described above.
                
        """
        if index < 0 or index >= self.__len__():
            raise IndexError(f'Index {index} is out of bounds for VideoDataset'
                             f' of size {self.__len__()}.')
    
        # find video of the frame
        i = len(self.offsets) - 1
        while (self.offsets[i] > index):
            i = i - 1
        
        filename = self.videos[i]
        filename = os.path.relpath(filename, self.root)

        splits = filename.split('.')
        video_format = splits[-1]
        video_name = '.'.join(splits[:-1])
        timestamp = float(self.video_timestamps[i][index - self.offsets[i]])
        return '%s-%.8fs-%s.png' % (video_name, timestamp, video_format)
