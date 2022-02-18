""" Video Dataset """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
from typing import List, Tuple
from fractions import Fraction
import threading
import weakref

from PIL import Image

import torch
import torchvision
from torchvision import datasets
from torchvision import io

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

if io._HAS_VIDEO_OPT:
    torchvision.set_video_backend('video_reader')

# @guarin 18.02.2022
# VideoLoader and VideoDataset multi-thread and multi-processing infos
# --------------------------------------------------------------------
# The VideoDataset class should be safe to use in multi-thread and 
# multi-processing settings. For the multi-processing setting it is assumed that
# a pytorch DataLoader is used. Multi-threading should not be use with the
# torchvision pyav video packend as pyav seems to be limited to a single thread.
# You will not see any speedups when using it from multiple threads!
# 
# The VideoLoader class is thread safe because it inherits from threading.local.
# When using it within a pytorch DataLoader a new instance should be created 
# in each process when using the torchvision video_reader backend, otherwise
# decoder errors can happen when iterating multiple times over the dataloader.
# This is specific to the video_reader backend and does not happen with the pyav
# backend.
# 
# In the VideoDataset class we avoid sharing VideoLoader instances between 
# workers by tracking the worker accessing the dataset. All VideoLoaders are 
# reset if a new worker accesses the dataset. Note that changes to the dataset 
# class by a worker are unique to that worker and not seen by other workers or 
# the main process.

class VideoLoader(threading.local):
    """Implementation of VideoLoader.

    The VideoLoader is a wrapper around the torchvision video interface. With
    the VideoLoader you can read specific frames or the next frames of a video.
    It automatically switches to the `video_loader` backend if available. Reading
    sequential frames is significantly faster since it uses the VideoReader 
    class from torchvision.

    The video loader automatically detects if you read out subsequent frames and
    will use the fast read method if possible. 

    Attributes:
        path:
            Root directory path.
        timestamps:
            Function that loads file at path.
        backend:
            Tuple of allowed extensions.
        transform:
            Function that takes a PIL image and returns transformed version
        target_transform:
            As transform but for targets
        is_valid_file:
            Used to check corrupt files

    Examples:
        >>> from torchvision import io
        >>>
        >>> # get timestamps
        >>> ts, fps = io.read_video_timestamps('myvideo.mp4', pts_unit = 'sec')
        >>>
        >>> # create a VideoLoader
        >>> video_loader = VideoLoader('myvideo.mp4', ts)
        >>>
        >>> # get frame at specific timestamp
        >>> frame = video_loader.read_frame(ts[21])
        >>>
        >>> # get next frame
        >>> frame = video_loader.read_frame()
    """
    def __init__(self, path: str, timestamps: List[float], backend: str = 'video_reader'):
        self.path = path
        self.timestamps = timestamps
        self.current_timestamp_idx = 0
        self.last_timestamp_idx = 0
        self.pts_unit='sec'
        self.backend = backend

        has_video_reader = io._HAS_VIDEO_OPT and hasattr(io, 'VideoReader')

        if has_video_reader and self.backend == 'video_reader':
            self.reader = io.VideoReader(path = self.path)
        else:
            self.reader = None
    
    def read_frame(self, timestamp = None):
        """Reads the next frame or from timestamp.

        If no timestamp is provided this method just returns the next frame from
        the video. This is significantly (up to 10x) faster if the `video_loader` 
        backend is available. If a timestamp is provided we first have to seek
        to the right position and then load the frame.
        
        Args:
            timestamp: Specific timestamp of frame in seconds or None (default: None)

        Returns:
            A PIL Image

        """
        if timestamp is not None:
            self.current_timestamp_idx = self.timestamps.index(timestamp)
        else:
            # no timestamp provided -> set current timestamp index to next frame
            if self.current_timestamp_idx < len(self.timestamps):
                self.current_timestamp_idx += 1

        if self.reader:
            if timestamp is not None:
                # Calling seek is slow. If we read next frame we can skip it!
                if self.timestamps.index(timestamp) != self.last_timestamp_idx + 1:
                    self.reader.seek(timestamp)

            # make sure we have the tensor in correct shape (we want H x W x C)
            frame = next(self.reader)['data'].permute(1,2,0)
            self.last_timestamp_idx = self.current_timestamp_idx

        else: # fallback on pyav
            if timestamp is None:
                # read next frame if no timestamp is provided
                timestamp = self.timestamps[self.current_timestamp_idx]
            frame, _, _ = io.read_video(self.path,
                                        start_pts=timestamp,
                                        end_pts=timestamp,
                                        pts_unit=self.pts_unit)    
            self.last_timestamp_idx = self.timestamps.index(timestamp)    
        
        
        if len(frame.shape) < 3:
            raise ValueError('Unexpected error during loading of frame')

        # sometimes torchvision returns multiple frames for one timestamp (bug?)
        if len(frame.shape) > 3 and frame.shape[0] > 1:
            frame = frame[0]

        # make sure we return a H x W x C tensor and not (1 x H x W x C)
        if len(frame.shape) == 4:
            frame = frame.squeeze()

        # convert to PIL image
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

    if extensions is None:
        if is_valid_file is None:
            ValueError('Both extensions and is_valid_file cannot be None')
        else:
            _is_valid_file = is_valid_file
    else:
        def is_valid_file_extension(filepath):
            return filepath.lower().endswith(extensions)
        if is_valid_file is None:
            _is_valid_file = is_valid_file_extension
        else:
            def _is_valid_file(filepath):
                return is_valid_file_extension(filepath) and is_valid_file(filepath)

    # find all video instances (no subdirectories)
    video_instances = []

    def on_error(error):
        raise error
    for root, _, files in os.walk(directory, onerror=on_error):

        for fname in files:
            # skip invalid files
            if not _is_valid_file(os.path.join(root, fname)):
                continue

            # keep track of valid files
            path = os.path.join(root, fname)
            video_instances.append(path)

    # get timestamps
    timestamps, fpss = [], []
    for instance in video_instances[:]: # video_instances[:] creates a copy

        if AV_AVAILABLE and torchvision.get_video_backend() == 'pyav':
            # This is a hacky solution to estimate the timestamps.
            # When using the video_reader this approach fails because the 
            # estimated timestamps are not correct.
            with av.open(instance) as av_video:
                stream = av_video.streams.video[0]

                # check if we can extract the video duration
                if not stream.duration:
                    print(f'Video {instance} has no timestamp and will be skipped...')
                    video_instances.remove(instance) # remove from original list (not copy)
                    continue # skip this broken video

                duration = stream.duration * stream.time_base
                fps = stream.base_rate
                n_frames = int(int(duration) * fps)

            timestamps.append([Fraction(i, fps) for i in range(n_frames)])
            fpss.append(fps)
        else:
            ts, fps = io.read_video_timestamps(instance, pts_unit=pts_unit)
            timestamps.append(ts)
            fpss.append(fps)

    # get frame offsets
    offsets = [len(ts) for ts in timestamps]
    offsets = [0] + offsets[:-1]
    for i in range(1, len(offsets)):
        offsets[i] = offsets[i-1] + offsets[i] # cumsum

    return video_instances, timestamps, offsets, fpss


class VideoDataset(datasets.VisionDataset):
    """Implementation of a video dataset.

    The VideoDataset allows random reads from a video file without extracting
    all frames beforehand. This is more storage efficient but is slower.

    Attributes:
        root:
            Root directory path.
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
                 extensions=None,
                 transform=None,
                 target_transform=None,
                 is_valid_file=None):
        
        super(VideoDataset, self).__init__(root,
                                           transform=transform,
                                           target_transform=target_transform)

        videos, video_timestamps, offsets, fps = _make_dataset(
            self.root, extensions, is_valid_file)
        
        if len(videos) == 0:
            msg = 'Found 0 videos in folder: {}\n'.format(self.root)
            if extensions is not None:
                msg += 'Supported extensions are: {}'.format(
                    ','.join(extensions))
            raise RuntimeError(msg)

        self.extensions = extensions
        self.backend = torchvision.get_video_backend()

        # Create initial set of video loaders which will be shared among
        # threads. When used with a dataloader a new video_loaders list will be
        # created in every worker process.
        self.video_loaders = [
            VideoLoader(video, timestamps, backend=self.backend) 
            for video, timestamps in zip(videos, video_timestamps)
        ]

        self.videos = videos
        self.video_timestamps = video_timestamps
        self._length = sum((
            len(ts) for ts in self.video_timestamps
        ))
        # offsets[i] indicates the index of the first frame of the i-th video.
        # e.g. for two videos of length 10 and 20, the offsets will be [0, 10].
        self.offsets = offsets
        self.fps = fps

        # Keep unique reference of dataloader worker. We need this to avoid
        # accidentaly sharing VideoLoader instances between workers.
        self._worker_ref = None

    def __getitem__(self, index):
        """Returns item at index.

        Finds the video of the frame at index with the help of the frame 
        offsets. Then, loads the frame from the video, applies the transforms,
        and returns the frame along with the index of the video (as target).

        For example, if there are two videos with 10 and 20 frames respectively
        in the input directory:

        Requesting the 5th sample returns the 5th frame from the first video and
        the target indicates the index of the source video which is 0.
        >>> dataset[5]
        >>> > <PIL Image>, 0

        Requesting the 20th sample returns the 10th frame from the second video
        and the target indicates the index of the source video which is 1.
        >>> dataset[20]
        >>> > <PIL Image>, 1

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

        # each sample belongs to a video, to load the sample at index, we need
        # to find the video to which the sample belongs and then read the frame
        # from this video on the disk.
        i = len(self.offsets) - 1
        while (self.offsets[i] > index):
            i = i - 1

        # find and return the frame as PIL image
        timestamp_idx = index - self.offsets[i]
        frame_timestamp = self.video_timestamps[i][timestamp_idx]
        video_loader = self._video_loader(i)
        sample = video_loader.read_frame(frame_timestamp)

        target = i
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        """Returns the number of samples (frames) in the dataset.

        This can be precomputed, because self.video_timestamps is only
        set in the __init__
        """
        return self._length

    def get_filename(self, index):
        """Returns a filename for the frame at index.

        The filename is created from the video filename, the frame number, and
        the video format. The frame number will be zero padded to make sure 
        all filenames have the same length and can easily be sorted.
        E.g. when retrieving a sample from the video
        `my_video.mp4` at frame 153, the filename will be:

        >>> my_video-153-mp4.png
    
        Args:
            index:
                Index of the frame to retrieve.

        Returns:
            The filename of the frame as described above.
                
        """
        if index < 0 or index >= self.__len__():
            raise IndexError(f'Index {index} is out of bounds for VideoDataset'
                             f' of size {self.__len__()}.')
    
        # each sample belongs to a video, to load the sample at index, we need
        # to find the video to which the sample belongs and then read the frame
        # from this video on the disk.
        i = len(self.offsets) - 1
        while (self.offsets[i] > index):
            i = i - 1

        # get filename of the video file
        video = self.videos[i]
        video_name, video_format = self._video_name_format(video)

        # get frame number
        frame_number = index - self.offsets[i]
        
        n_frames = self._video_frame_count(i)
        zero_padding = len(str(n_frames))

        return self._format_filename(
            video_name=video_name,
            video_format=video_format,
            frame_number=frame_number,
            zero_padding=zero_padding,
        )

    def get_filenames(self) -> List[str]:
        """Returns a list filenames for all frames in the dataset.
        
        """
        filenames = []
        for i, video in enumerate(self.videos):
            video_name, video_format = self._video_name_format(video)
            n_frames = self._video_frame_count(i)

            zero_padding = len(str(n_frames))
            for frame_number in range(n_frames):
                filenames.append(
                    self._format_filename(
                        video_name=video_name,
                        frame_number=frame_number,
                        video_format=video_format,
                        zero_padding=zero_padding,
                    )
                )
        return filenames

    def _video_frame_count(self, video_index: int) -> int:
        """Returns the number of frames in the video with the given index.
        
        """
        if video_index < len(self.offsets) - 1:
            n_frames = self.offsets[video_index+1] - self.offsets[video_index]
        else:
            n_frames = len(self) - self.offsets[video_index]
        return n_frames

    def _video_name_format(self, video_filename: str) -> Tuple[str, str]:
        """Extracts name and format from the filename of the video.

        Returns:
            A (video_name, video_format) tuple where video_name is the filename
            relative to self.root and video_format is the file extension, for 
            example 'mp4'.

        """
        video_filename = os.path.relpath(video_filename, self.root)
        splits = video_filename.split('.')
        video_format = splits[-1]
        video_name = '.'.join(splits[:-1])
        return video_name, video_format

    def _format_filename(
        self,
        video_name: str, 
        frame_number: int, 
        video_format: str, 
        zero_padding: int = 8, 
        extension: str = 'png'
    ) -> str:
        return f'{video_name}-{frame_number:0{zero_padding}}-{video_format}.{extension}'

    def _video_loader(self, video_index: int) -> VideoLoader:
        """Returns a video loader unique to the current dataloader worker."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Use a weakref instead of worker_info.id as the worker id is reused
            # by different workers across epochs.
            worker_ref = weakref.ref(worker_info)
            if worker_ref != self._worker_ref:
                print(f'changing worker_ref {self._worker_ref} -> {worker_ref}')
                self._worker_ref = worker_ref
                # Initialize empty video loaders for this worker.
                # Note that changes to self.video_loaders are not propagated
                # across worker processes, so this is safe.
                self.video_loaders = [None] * len(self.videos)
        
        video_loader = self.video_loaders[video_index]
        if video_loader is None:
            # The worker has not accessed this video before.
            video = self.videos[video_index]
            timestamps = self.video_timestamps[video_index]
            video_loader = VideoLoader(video, timestamps, backend=self.backend)
            self.video_loaders[video_index] = video_loader
        
        return video_loader
