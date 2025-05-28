from __future__ import annotations

import concurrent.futures
import os
import pathlib
import shutil
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol

import requests
import tqdm

from lightly.api import utils


def download_and_write_file(
    url: str,
    output_path: str,
    session: requests.Session | None = None,
    retry_fn: Callable[..., Any] = utils.retry,
    request_kwargs: dict[str, Any] | None = None,
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
    with retry_fn(req.get, url=url, **request_kwargs) as response:  # type: ignore[attr-defined]
        response.raise_for_status()
        with open(out_path, "wb") as file:
            shutil.copyfileobj(response.raw, file)


def download_and_write_all_files(
    file_infos: list[tuple[str, str]],
    output_dir: str,
    max_workers: int | None = None,
    verbose: bool = False,
    retry_fn: Callable[..., Any] = utils.retry,
    request_kwargs: dict[str, Any] | None = None,
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
        file_info: tuple[str, str],
        output_dir: str,
        lock: threading.Lock,
        sessions: dict[int, requests.Session],
        retry_fn: Callable[..., Any],
        request_kwargs: dict[str, Any] | None = None,
    ) -> None:
        filename, url = file_info
        output_path = os.path.join(output_dir, filename)
        thread_id = threading.get_ident()

        lock.acquire()
        session = sessions.get(thread_id)
        if session is None:
            session = requests.Session()
            sessions[thread_id] = session
        lock.release()

        download_and_write_file(url, output_path, session, retry_fn, request_kwargs)

    # dict where every thread stores its requests.Session
    sessions: dict[int, requests.Session] = dict()
    # use lock because sessions dict is shared between threads
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_file_info = {
            executor.submit(
                thread_download_and_write,
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
