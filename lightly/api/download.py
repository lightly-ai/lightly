import concurrent.futures
import os
import pathlib
import shutil
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import requests
import tqdm

from lightly.api import retry_utils


def download_and_write_file(
    url: str,
    output_path: str,
    session: requests.Session = None,
    retry_fn: Callable = retry_utils.retry,
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
    request_kwargs.setdefault("stream", True)
    request_kwargs.setdefault("timeout", 10)
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
    retry_fn: Callable = retry_utils.retry,
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
