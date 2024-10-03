from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence
from urllib import parse

from lightly.data import _helpers


def get_server(
    paths: Sequence[Path],
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    """Returns an HTTP server that serves a local datasource.

    Args:
        paths:
            List of paths to serve.
        host:
            Host to serve the datasource on.
        port:
            Port to serve the datasource on.

    Examples:
        >>> from lightly.api import serve
        >>> from pathlib import Path
        >>> serve(
        >>>    paths=[Path("/input_mount), Path("/lightly_mount)],
        >>>    host="localhost",
        >>>    port=3456,
        >>> )

    """

    class _LocalDatasourceRequestHandler(SimpleHTTPRequestHandler):
        def translate_path(self, path: str) -> str:
            return _translate_path(path=path, directories=paths)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def send_response_only(self, code: int, message: str | None = None) -> None:
            super().send_response_only(code, message)
            self.send_header(
                "Cache-Control", "no-store, must-revalidate, no-cache, max-age=-1"
            )
            self.send_header("Expires", "0")

    return ThreadingHTTPServer((host, port), _LocalDatasourceRequestHandler)


def validate_input_mount(input_mount: Path) -> None:
    """Validates that the input mount is a directory and contains files."""
    input_mount = input_mount.resolve()
    if not input_mount.exists():
        raise ValueError(
            f"Path for 'input_mount' argument '{input_mount}' does not exist."
        )
    if not input_mount.is_dir():
        raise ValueError(
            f"Path for 'input_mount' argument '{input_mount}' is not a directory."
        )
    if not _dir_contains_image_or_video(path=input_mount):
        raise ValueError(
            f"Path for 'input_mount' argument '{input_mount}' does not contain any "
            "images or videos. Please verify that this is the correct directory. See "
            "our docs on lightly-serve for more information: "
            "https://docs.lightly.ai/docs/local-storage#optional-after-run-view-local-data-in-lightly-platform"
        )


def validate_lightly_mount(lightly_mount: Path) -> None:
    lightly_mount = lightly_mount.resolve()
    """Validates that the Lightly mount is a directory."""
    if not lightly_mount.exists():
        raise ValueError(
            f"Path for 'lightly_mount' argument '{lightly_mount}' does not exist."
        )
    if not lightly_mount.is_dir():
        raise ValueError(
            f"Path for 'lightly_mount' argument '{lightly_mount}' is not a directory."
        )


def _dir_contains_image_or_video(path: Path) -> bool:
    extensions = set(_helpers.IMG_EXTENSIONS + _helpers.VIDEO_EXTENSIONS)
    return any(
        p for p in path.rglob("**/*") if p.is_file() and p.suffix.lower() in extensions
    )


def _translate_path(path: str, directories: Sequence[Path]) -> str:
    """Translates a relative path to a file in the local datasource.

    Tries to resolve the relative path to a file in the first directory
    and serves it if it exists. Otherwise, it tries to resolve the relative
    path to a file in the second directory and serves it if it exists, etc.

    Args:
        path:
            Relative path to a file in the local datasource.
        directories:
            List of directories to search for the file.


    Returns:
        Absolute path to the file in the local datasource or an empty string
        if the file doesn't exist.

    """
    path = parse.unquote(path)
    stripped_path = path.lstrip("/")
    for directory in directories:
        if (directory / stripped_path).exists():
            return str(directory / stripped_path)
    return ""  # Not found.
