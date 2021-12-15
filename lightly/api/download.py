import concurrent.futures
import os
import pathlib
import shutil
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Iterable, Union

import av
import requests
import tqdm
from lightly.api import utils
import PIL


def download_image(url: str, session: requests.Session = None) -> PIL.Image.Image:
    """Downloads an image from an url.

    Args:
        url: 
            Url where image is stored.
        session: 
            Make request using the given session.

    Returns:
        The downloaded image.

    """
    req = requests if session is None else session
    response = req.get(url, stream=True)
    return PIL.Image.open(response.raw)


def download_all_video_frames(
    url: str, as_pil_image: int = True
) -> Iterable[Union[PIL.Image.Image, av.VideoFrame]]:
    """Lazily retrieves all frames from a video.

    Args:
        url: 
            Url where video is stored.
        as_pil_image: 
            Whether to return the frame as PIL.Image.

    Returns:
        A generator that loads and returns a single frame per step.

    """
    container = av.open(url)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    for frame in container.decode(stream):
        if as_pil_image:
            yield frame.to_image()
        else:
            yield frame
    container.close()


def download_video_frame(
    url: str, timestamp: float, as_pil_image: int = True
) -> Union[PIL.Image.Image, av.VideoFrame]:
    """Retrieves a specific frame from a video stored at `url`.

    Args:
        url: 
            The url where the video is stored.
        timestamp:
            Timestamp in seconds from the start of the video at
            which the frame should be retrieved.
        as_pil_image:
            Whether to return the frame as PIL.Image.

    Returns:
        The downloaded video frame

    """
    if timestamp < 0:
        raise ValueError(f"Negative timestamp is not allowed: {timestamp}")

    container = av.open(url)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    offset = int(timestamp / stream.time_base)
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
    """Downloads a file from url and saves it to disk

    Args:
        url: 
            Url of the file to download.
        output_path: 
            Where to store the file, including filename and extension.
        session: 
            Make request using the given session.
    """
    req = requests if session is None else session
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with req.get(url, stream=True) as response:
        with open(out_path, "wb") as file:
            shutil.copyfileobj(response.raw, file)


def download_and_write_all_files(
    file_infos: List[tuple],
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
            Maximum number of workers
            If `None` the number of workers is chosen based
            on the number of available cores.
        verbose:
            Shows progress bar if set to `True`.

    """

    def thread_download_and_write(
        file_info: tuple, output_dir: str, lock: threading.Lock, sessions: dict
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
