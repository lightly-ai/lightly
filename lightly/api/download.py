import concurrent.futures
import os
import pathlib
import shutil
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple, Union

import PIL
import requests
import tqdm
from lightly.api import utils

try:
    import av
except ModuleNotFoundError:
    av = ModuleNotFoundError(
        "PyAV is not installed on your system. Please install it to use the video"
        "functionalities. See https://github.com/mikeboers/PyAV#installation for"
        "installation instructions."
    )

def _check_av_available() -> None:
    if isinstance(av, Exception):
        raise av

def download_image(url: str, session: requests.Session = None) -> PIL.Image.Image:
    """Downloads an image from a url.

    Args:
        url: 
            The url where the image is downloaded from.
        session: 
            Session object to persist certain parameters across requests.

    Returns:
        The downloaded image.

    """
    req = requests if session is None else session
    response = utils.retry(req.get, url=url, stream=True)
    return PIL.Image.open(response.raw)


def download_all_video_frames(
    url: str, 
    timestamp: Union[int, None] = None,
    as_pil_image: int = True, 
    thread_type: av.codec.context.ThreadType = av.codec.context.ThreadType.AUTO,
    video_channel: int = 0,
) -> Iterable[Union[PIL.Image.Image, av.VideoFrame]]:
    """Lazily retrieves all frames from a video stored at the given url.

    Args:
        url: 
            The url where video is downloaded from.
        timestamp:
            Timestamp in pts from the start of the video from which the frame 
            download should start. See https://pyav.org/docs/develop/api/time.html#time
            for details on pts.
        as_pil_image: 
            Whether to return the frame as PIL.Image.
        thread_type:
            Which multithreading method to use for decoding the video.
            See https://pyav.org/docs/stable/api/codec.html#av.codec.context.ThreadType
            for details.
        video_channel:
            The video channel from which frames are loaded.

    Returns:
        A generator that loads and returns a single frame per step.

    """
    _check_av_available()
    timestamp = 0 if timestamp is None else timestamp
    if timestamp < 0:
        raise ValueError(f"Negative timestamp is not allowed: {timestamp}")

    with utils.retry(av.open, url) as container:
        stream = container.streams.video[video_channel]
        stream.thread_type = thread_type

        duration = stream.duration
        start_time = stream.start_time
        if (duration is not None) and (start_time is not None):
            end_time = duration + start_time
            if timestamp > end_time:
                raise ValueError(
                    f"Timestamp ({timestamp} pts) exceeds maximum video timestamp "
                    f"({end_time} pts)."
                )
        # seek to last keyframe before the timestamp
        container.seek(timestamp, any_frame=False, backward=True, stream=stream)
        
        frame = None
        for frame in container.decode(stream):
            # advance from keyframe until correct timestamp is reached
            if frame.pts < timestamp:
                continue
            # yield next frame
            if as_pil_image:
                yield frame.to_image()
            else:
                yield frame


def download_video_frame(
    url: str, 
    timestamp: int, 
    as_pil_image: int = True,
    thread_type: av.codec.context.ThreadType = av.codec.context.ThreadType.AUTO,
    video_channel: int = 0,
) -> Union[PIL.Image.Image, av.VideoFrame, None]:
    """Retrieves a specific frame from a video stored at the given url.

    Finds the first frame in the video that has a timestamp equal or larger than
    the timestamp argument.

    Args:
        url: 
            The url where the video is downloaded from.
        timestamp:
            Timestamp in pts from the start of the video at which the frame 
            should be retrieved. See https://pyav.org/docs/develop/api/time.html#time
            for details on pts.
        as_pil_image:
            Whether to return the frame as PIL.Image.
        thread_type:
            Which multithreading method to use for decoding the video.
            See https://pyav.org/docs/stable/api/codec.html#av.codec.context.ThreadType
            for details.
        video_channel:
            The video channel from which frames are loaded.

    Returns:
        The downloaded video frame or None if no frame could be found.

    """
    _check_av_available()
    if timestamp < 0:
        raise ValueError(f"Negative timestamp is not allowed: {timestamp}")

    with utils.retry(av.open, url) as container:
        stream = container.streams.video[video_channel]
        stream.thread_type = thread_type
        
        duration = stream.duration
        start_time = stream.start_time
        if (duration is not None) and (start_time is not None):
            end_time = duration + start_time
            if timestamp > end_time:
                raise ValueError(
                    f"Timestamp ({timestamp} pts) exceeds maximum video timestamp "
                    f"({end_time} pts)."
                )
        # seek to last keyframe before the timestamp
        container.seek(timestamp, any_frame=False, backward=True, stream=stream)
        # advance from keyframe until correct timestamp is reached
        frame = None
        for frame in container.decode(stream):
            if frame.pts >= timestamp:
                break

    if as_pil_image:
        return frame.to_image()
    return frame


def download_and_write_file(
    url: str, output_path: str, session: requests.Session = None
) -> None:
    """Downloads a file from a url and saves it to disk

    Args:
        url: 
            Url of the file to download.
        output_path: 
            Where to store the file, including filename and extension.
        session: 
            Session object to persist certain parameters across requests.
    """
    req = requests if session is None else session
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with utils.retry(req.get, url=url, stream=True) as response:
        with open(out_path, "wb") as file:
            shutil.copyfileobj(response.raw, file)


def download_and_write_all_files(
    file_infos: List[Tuple[str, str]],
    output_dir: str,
    max_workers: int = None,
    verbose: bool = False,
) -> None:
    """Downloads all files and writes them to disk.

    Args:
        file_infos:
            List containing (filename, url) tuples.
        output_dir:
            Output directory where files will stored in.
        max_workers:
            Maximum number of workers. If `None` the number of workers is chosen 
            based on the number of available cores.
        verbose:
            Shows progress bar if set to `True`.

    """

    def thread_download_and_write(
        file_info: Tuple[str, str], 
        output_dir: str, 
        lock: threading.Lock, 
        sessions: Dict[str, requests.Session]
    ):
        filename, url = file_info
        output_path = os.path.join(output_dir, filename)
        thread_id = threading.get_ident()

        lock.acquire()
        session = sessions.get(thread_id)
        if session is None:
            session = requests.Session()
            sessions[thread_id] = session
        lock.release()

        download_and_write_file(url, output_path, session)

    # retry download if failed
    def job(*args):
        utils.retry(thread_download_and_write, *args)

    # dict where every thread stores its requests.Session
    sessions = dict()
    # use lock because sessions dict is shared between threads
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_file_info = {
            executor.submit(
                job, file_info, output_dir, lock, sessions
            ): file_info
            for file_info in file_infos
        }
        futures = concurrent.futures.as_completed(futures_to_file_info)
        if verbose:
            futures = tqdm.tqdm(futures)
        for future in futures:
            filename, url = futures_to_file_info[future]
            try:
                future.result()
            except Exception as ex:
                warnings.warn(f"Could not download {filename} from {url}")

def video_frame_count(
    url: str,
    video_channel: int = 0,
    thread_type: av.codec.context.ThreadType = av.codec.context.ThreadType.AUTO,
    ignore_metadata: bool = False,
) -> Optional[int]:
    """Returns the number of frames in the video from the given url.

    The video is only decoded if no information about the number of frames is
    stored in the video metadata.
    
    Args:
        url:
            The url of the video.
        video_channel:
            The video stream channel from which to find the number of frames.
        thread_type:
            Which multithreading method to use for decoding the video.
            See https://pyav.org/docs/stable/api/codec.html#av.codec.context.ThreadType
            for details.
        ignore_metadata:
            If True, frames are counted by iterating through the video instead
            of relying on the video metadata.

    Returns:
        The number of frames in the video. Can be None if the video could not be
        decoded.

    """
    with av.open(url) as container:
        stream = container.streams.video[video_channel]
        num_frames = 0 if ignore_metadata else stream.frames
        # If number of frames not stored in the video file we have to decode all
        # frames and count them.
        if num_frames == 0:
            stream.thread_type = thread_type
            for _ in container.decode(stream):
                num_frames += 1
    return num_frames

def all_video_frame_counts(
    urls: List[str],
    max_workers: int = None,
    video_channel: int = 0,
    thread_type: av.codec.context.ThreadType = av.codec.context.ThreadType.AUTO,
    ignore_metadata: bool = False,
) -> List[Optional[int]]:
    """Finds the number of frames in the videos at the given urls.

    Videos are only decoded if no information about the number of frames is
    stored in the video metadata.

    Args:
        urls:
            A list of video urls.
        max_workers:
            Maximum number of workers. If `None` the number of workers is chosen 
            based on the number of available cores.
        video_channel:
            The video stream channel from which to find the number of frames.
        thread_type:
            Which multithreading method to use for decoding the video.
            See https://pyav.org/docs/stable/api/codec.html#av.codec.context.ThreadType
            for details.
        ignore_metadata:
            If True, frames are counted by iterating through the video instead
            of relying on the video metadata.

    Returns:
        A list with the number of frames per video. Contains None for all videos
        that could not be decoded.

    """

    def job(url):
        try:
            return utils.retry(
                video_frame_count, 
                url=url,
                video_channel=video_channel,
                thread_type=thread_type,
                ignore_metadata=ignore_metadata,
            )
        except RuntimeError:
            return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(job, urls))
