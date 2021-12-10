import concurrent.futures
import os
import pathlib
import shutil
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List

import cv2
import requests
import tqdm
from lightly.api import utils
from PIL import Image


def cv2_to_pil(image):
    """Convert cv2 image to PIL.Image"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def download_image(url: str, session: requests.Session = None):
    """Downloads an image from an url

    Args:
        url (str):
            Url where image is stored
        session (requests.Session):
            Make request using the given session

    Returns:
        PIL.Image
    """
    req = requests if session is None else session
    response = req.get(url, stream=True)
    return Image.open(response.raw)


def download_all_video_frames(url: str, as_pil_image: int = True):
    """Lazily retrieves all frames from a video

    Args:
        url (str):
            Url where video is stored
        as_pil_image (bool):
            Whether to return the frame as PIL.Image

    Returns:
        generator:
            Loads and returns a single frame per step.
            Frames are of type PIL.Image if `as_pil_image` is `True`,
            otherwise a np.array is returned

    """
    video = cv2.VideoCapture(url)
    while True:
        frame_exists, frame = video.read()
        if not frame_exists:
            break
        if as_pil_image:
            yield cv2_to_pil(frame)
        else:
            yield frame
    video.release()


def download_video_frame(url: str, frame_index: int, as_pil_image: int = True):
    """Retrieves a specific frame from a video stored at `url`

    Warning: This is pretty slow

    Args:
        url (str):
            The url where the video is stored
        frame_index (int):
            Zero based index of the frame to retrieve
        as_pil_image (bool):
            Whether to return the frame as PIL.Image

    Returns:
        PIL.Image: If `as_pil_image` is `True`
        np.array: If `as_pil_image` is `False`
    """
    if frame_index < 0:
        raise IndexError(f"Negative frame_index: {frame_index}")

    video = cv2.VideoCapture(url)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_index >= num_frames:
        raise IndexError(
            f"frame_index too large: f{frame_index}, video has {num_frames} frames"
        )

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    frame_exists, frame = video.read()
    video.release()

    if as_pil_image:
        return cv2_to_pil(frame)
    return frame


def download_and_write_file(
    url: str, output_path: str, session: requests.Session = None
):
    """Downloads a file from url and saves it to disk

    Args:
        url (str):
            Url of the file to download
        output_path (str):
            Where to store the file, including filename and extension
        session (str):
            Make request using the given session
    """
    req = requests if session is None else session
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with req.get(url, stream=True) as response:
        with open(output_path, "wb") as file:
            shutil.copyfileobj(response.raw, file)


def download_and_write_all_files(
    file_infos: List[tuple],
    output_dir: str,
    max_workers: int = None,
    verbose: bool = False,
):
    """Downloads all files and writes them to disk.

    Args:
        file_infos (list(tuple)):
            List containing (filename, url) tuples
        output_dir (str):
            Output directory where files will stored in
        max_workers (int):
            Maximum number of workers
            If `None` the number of workers is chosen based
            on the number of available cores
        verbose (bool):
            Shows progress bar if set to `True`
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
