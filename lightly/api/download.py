import concurrent.futures
import os
import pathlib
import shutil
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

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

def download_image(
    url: str, 
    session: requests.Session = None,
    retry_fn: Callable = utils.retry,
    request_kwargs: Optional[Dict] = None,
) -> PIL.Image.Image:
    """Downloads an image from a url.

    Args:
        url: 
            The url where the image is downloaded from.
        session: 
            Session object to persist certain parameters across requests.
        retry_fn:
            Retry function that handles failed downloads.
        request_kwargs:
            Additional parameters passed to requests.get().

    Returns:
        The downloaded image.

    """
    request_kwargs = request_kwargs or {}
    request_kwargs.setdefault('stream', True)
    request_kwargs.setdefault('timeout', 10)
    
    def load_image(url, req, request_kwargs):
        with req.get(url=url, **request_kwargs) as response:
            response.raise_for_status()
            image = PIL.Image.open(response.raw)
            image.load()
        return image

    req = requests if session is None else session
    image = retry_fn(load_image, url, req, request_kwargs)
    return image

if not isinstance(av, ModuleNotFoundError):

    def download_all_video_frames(
        url: str,
        timestamp: Union[int, None] = None,
        as_pil_image: int = True,
        thread_type: av.codec.context.ThreadType = av.codec.context.ThreadType.AUTO,
        video_channel: int = 0,
        retry_fn: Callable = utils.retry,
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
            retry_fn:
                Retry function that handles errors when opening the video container.

        Returns:
            A generator that loads and returns a single frame per step.

        """
        _check_av_available()
        timestamp = 0 if timestamp is None else timestamp
        if timestamp < 0:
            raise ValueError(f"Negative timestamp is not allowed: {timestamp}")

        with retry_fn(av.open, url) as container:
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


    def download_video_frame(url: str, timestamp: int, *args, **kwargs
                             ) -> Union[PIL.Image.Image, av.VideoFrame, None]:
        """
        Wrapper around download_video_frames_at_timestamps
        for downloading only a single frame.
        """
        frames = download_video_frames_at_timestamps(
            url, timestamps=[timestamp], *args, **kwargs
        )
        frames = list(frames)
        return frames[0]


    def video_frame_count(
        url: str,
        video_channel: int = 0,
        thread_type: av.codec.context.ThreadType = av.codec.context.ThreadType.AUTO,
        ignore_metadata: bool = False,
        retry_fn: Callable = utils.retry,
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
        with retry_fn(av.open, url) as container:
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
        retry_fn: Callable = utils.retry,
        exceptions_indicating_empty_video: Tuple[Type[BaseException], ...] = (RuntimeError,),
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
            retry_fn:
                Retry function that handles errors when loading a video.
            exceptions_indicating_empty_video:
                If an exception in exceptions_indicating_empty_video is raised,
                the video is considered as empty and None is returned as number
                of frames for the video.

        Returns:
            A list with the number of frames per video. Contains None for all videos
            that could not be decoded.

        """

        def job(url):
            try:
                return retry_fn(
                    video_frame_count,
                    url=url,
                    video_channel=video_channel,
                    thread_type=thread_type,
                    ignore_metadata=ignore_metadata,
                    retry_fn=retry_fn,
                )
            except exceptions_indicating_empty_video:
                return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(job, urls))



    def download_video_frames_at_timestamps(
            url: str,
            timestamps: List[int],
            as_pil_image: int = True,
            thread_type: av.codec.context.ThreadType = av.codec.context.ThreadType.AUTO,
            video_channel: int = 0,
            seek_to_first_frame: bool = True,
            retry_fn: Callable = utils.retry,
        ) -> Iterable[Union[PIL.Image.Image, av.VideoFrame]]:
            """Lazily retrieves frames from a video at a specific timestamp stored at the given url.

            Args:
                url:
                    The url where video is downloaded from.
                timestamps:
                    Timestamps in pts from the start of the video. The images
                    at these timestamps are returned.
                    The timestamps must be strictly monotonically ascending.
                    See https://pyav.org/docs/develop/api/time.html#time
                    for details on pts.
                as_pil_image:
                    Whether to return the frame as PIL.Image.
                thread_type:
                    Which multithreading method to use for decoding the video.
                    See https://pyav.org/docs/stable/api/codec.html#av.codec.context.ThreadType
                    for details.
                video_channel:
                    The video channel from which frames are loaded.
                seek_to_first_frame:
                    Boolean indicating whether to seek to the first frame.
                retry_fn:
                    Retry function that handles errors when opening the video container.

            Returns:
                A generator that loads and returns a single frame per step.

            """
            _check_av_available()

            if len(timestamps) == 0:
                return []

            if any(
                    timestamps[i+1] <= timestamps[i]
                    for i
                    in range(len(timestamps) - 1)
            ):
                raise ValueError("The timestamps must be sorted "
                                 "strictly monotonically ascending, but are not.")
            min_timestamp = timestamps[0]
            max_timestamp = timestamps[-1]

            if min_timestamp < 0:
                raise ValueError(f"Negative timestamp is not allowed: {min_timestamp}")

            with retry_fn(av.open, url) as container:
                stream = container.streams.video[video_channel]
                stream.thread_type = thread_type

                # Heuristic to find out if a timestamp is too big.
                # However, this heuristic is very bad, as timestamps
                # may start at any offset and even reset in the middle
                # See https://stackoverflow.com/questions/10570685/avcodec-pts-timestamps-not-starting-at-0
                duration = stream.duration
                start_time = stream.start_time
                if (duration is not None) and (start_time is not None):
                    end_time = duration + start_time
                    if max_timestamp > end_time:
                        raise ValueError(
                            f"Timestamp ({max_timestamp} pts) exceeds maximum video timestamp "
                            f"({end_time} pts).")

                if seek_to_first_frame:
                    # seek to last keyframe before the min_timestamp
                    container.seek(
                        min_timestamp,
                        any_frame=False,
                        backward=True,
                        stream=stream
                    )

                index_timestamp = 0
                for frame in container.decode(stream):

                    # advance from keyframe until correct timestamp is reached
                    if frame.pts > timestamps[index_timestamp]:

                        # dropped frames!
                        break

                    # it's ok to check by equality because timestamps are ints
                    if frame.pts == timestamps[index_timestamp]:

                        # yield next frame
                        if as_pil_image:
                            yield frame.to_image()
                        else:
                            yield frame

                        # update the timestamp
                        index_timestamp += 1

                    if index_timestamp >= len(timestamps):
                        return

            leftovers = timestamps[index_timestamp:]

            # sometimes frames are skipped when we seek to the first frame
            # let's retry downloading these frames without seeking
            retry_skipped_timestamps = seek_to_first_frame
            if retry_skipped_timestamps:
                warnings.warn(
                    f'Timestamps {leftovers} could not be decoded! Retrying from the start...'
                )
                frames = download_video_frames_at_timestamps(
                    url,
                    leftovers,
                    as_pil_image=as_pil_image,
                    thread_type=thread_type,
                    video_channel=video_channel,
                    seek_to_first_frame=False,
                    retry_fn=retry_fn,
                )
                for frame in frames:
                    yield frame
                return

            raise RuntimeError(
                f'Timestamps {leftovers} in video {url} could not be decoded!'
            )


def download_and_write_file(
    url: str, output_path: str, 
    session: requests.Session = None,
    retry_fn: Callable = utils.retry,
    request_kwargs: Optional[Dict] = None,
) -> None:
    """Downloads a file from a url and saves it to disk

    Args:
        url:
            Url of the file to download.
        output_path:
            Where to store the file, including filename and extension.
        session:
            Session object to persist certain parameters across requests.
        retry_fn:
            Retry function that handles failed downloads.
        request_kwargs:
            Additional parameters passed to requests.get().
    """
    request_kwargs = request_kwargs or {}
    request_kwargs.setdefault('stream', True)
    request_kwargs.setdefault('timeout', 10)
    req = requests if session is None else session
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with retry_fn(req.get, url=url, **request_kwargs) as response:
        response.raise_for_status()
        with open(out_path, "wb") as file:
            shutil.copyfileobj(response.raw, file)


def download_and_write_all_files(
    file_infos: List[Tuple[str, str]],
    output_dir: str,
    max_workers: int = None,
    verbose: bool = False,
    retry_fn: Callable = utils.retry,
    request_kwargs: Optional[Dict] = None,
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
        retry_fn:
            Retry function that handles failed downloads.
        request_kwargs:
            Additional parameters passed to requests.get().

    """

    def thread_download_and_write(
        file_info: Tuple[str, str],
        output_dir: str,
        lock: threading.Lock,
        sessions: Dict[str, requests.Session],
        **kwargs,
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

        download_and_write_file(url, output_path, session, **kwargs)

    # retry download if failed
    def job(**kwargs):
        retry_fn(thread_download_and_write, **kwargs)

    # dict where every thread stores its requests.Session
    sessions = dict()
    # use lock because sessions dict is shared between threads
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_file_info = {
            executor.submit(
                job, 
                file_info=file_info, 
                output_dir=output_dir, 
                lock=lock, 
                sessions=sessions, 
                retry_fn=retry_fn, 
                request_kwargs=request_kwargs,
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

def download_prediction_file(
    url: str,
    session: requests.Session = None,
    request_kwargs: Optional[Dict] = None,
) -> Union[Dict, None]:
    """Same as download_json_file. Keep this for backwards compatability.

    Args:
        url:
            Url of the file to download.
        session:
            Session object to persist certain parameters across requests.
        request_kwargs:
            Additional parameters passed to requests.get().

    Returns the content of the json file as dictionary or None.

    """
    return download_json_file(url, session=session, request_kwargs=request_kwargs)

def download_json_file(
    url: str,
    session: requests.Session = None,
    request_kwargs: Optional[Dict] = None,
) -> Union[Dict, None]:
    """Downloads a json file from the provided read-url.

    Args:
        url:
            Url of the file to download.
        session: 
            Session object to persist certain parameters across requests.
        request_kwargs:
            Additional parameters passed to requests.get().

    Returns the content of the json file as dictionary or None.

    """
    request_kwargs = request_kwargs or {}
    request_kwargs.setdefault('stream', True)
    request_kwargs.setdefault('timeout', 10)
    req = requests if session is None else session
    response = req.get(url, **request_kwargs)

    if response.status_code < 200 or response.status_code >= 300:
        return None # the file doesn't exist!

    return response.json()
