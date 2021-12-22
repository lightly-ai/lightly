import concurrent.futures
import os
import pathlib
import shutil
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Iterable, Union, Tuple, Dict

import requests
import tqdm
from lightly.api import utils
import PIL

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
    response = req.get(url, stream=True)
    return PIL.Image.open(response.raw)


def download_all_video_frames(
    url: str, 
    as_pil_image: int = True, 
    thread_type: av.codec.context.ThreadType = av.codec.context.ThreadType.AUTO,
    video_channel: int = 0,
) -> Iterable[Union[PIL.Image.Image, av.VideoFrame]]:
    """Lazily retrieves all frames from a video stored at the given url.

    Args:
        url: 
            The url where video is downloaded from.
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
    container = av.open(url)
    stream = container.streams.video[video_channel]
    stream.thread_type = thread_type
    for frame in container.decode(stream):
        if as_pil_image:
            yield frame.to_image()
        else:
            yield frame
    container.close()


def download_video_frame(
    url: str, 
    timestamp: float, 
    as_pil_image: int = True,
    thread_type: av.codec.context.ThreadType = av.codec.context.ThreadType.AUTO,
    video_channel: int = 0,
    time_unit: str = 'sec',
) -> Union[PIL.Image.Image, av.VideoFrame]:
    """Retrieves a specific frame from a video stored at the given url.

    Args:
        url: 
            The url where the video is downloaded from.
        timestamp:
            Timestamp in time_unit from the start of the video at
            which the frame should be retrieved.
        as_pil_image:
            Whether to return the frame as PIL.Image.
        thread_type:
            Which multithreading method to use for decoding the video.
            See https://pyav.org/docs/stable/api/codec.html#av.codec.context.ThreadType
            for details.
        video_channel:
            The video channel from which frames are loaded.
        time_unit:
            One of 'sec' or 'pts'. Determines how timestamp is interpreted.

    Returns:
        The downloaded video frame.

    """
    _check_av_available()
    if timestamp < 0:
        raise ValueError(f"Negative timestamp is not allowed: {timestamp}")
    if time_unit not in ('sec', 'pts'):
        raise ValueError(f"time_unit must be 'sec' or 'pts' but is {time_unit}")

    container = av.open(url)
    stream = container.streams.video[video_channel]
    stream.thread_type = thread_type
    
    if time_unit == 'sec':
        offset = int(timestamp / stream.time_base)
    else:
        offset = int(timestamp)

    duration = stream.duration
    if offset >= duration:
        duration_seconds = duration * stream.time_base
        raise ValueError(
            f"Timestamp ({timestamp}) larger than"
            f"video duration ({duration_seconds})"
        )
    # seek to last keyframe before the timestamp
    container.seek(offset, any_frame=False, backward=True, stream=stream)
    # advance from keyframe until correct offset is reached
    frame = None
    for frame in container.decode(stream):
        if frame.pts > offset:
            break
    container.close()
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
    with req.get(url, stream=True) as response:
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
) -> int:
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

    """
    container = av.open(url)
    stream = container.streams.video[video_channel]
    num_frames = stream.frames
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
) -> Tuple[int, List[int]]:
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

    Returns:
        A (total_frames, frame_counts) tuple where total_frames is the sum of 
        all frames in all videos and frame_counts is a list with the number of
        frames per video.

    Raises:
        RuntimeError:
            If not all frame counts can be found for all videos.

    """

    def job(url):
        return utils.retry(
            video_frame_count, 
            url=url,
            video_channel=video_channel,
            thread_type=thread_type,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frame_counts = list(executor.map(job, urls))
        num_failed = 0
        for count, url in zip(frame_counts, urls):
            if count is None:
                num_failed += 1
                warnings.warn(
                    f"Could not find the number of frames for video with "
                    f"url: {url}"
                )

        if num_failed > 0:
            raise RuntimeError(
                f"Could not find the number of frames for {num_failed} videos!"
            )

        total_frames = sum(frame_counts)
        return total_frames, frame_counts
