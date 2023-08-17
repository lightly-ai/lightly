import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Sequence, Type


def get_server(
    root_dir: Path,
    input_path: str,
    lightly_path: str,
    host: str,
    port: int,
):
    """Returns an HTTP server that serves a local datasource.

    Args:
        root_dir:
            Root directory of the local datasource.
        input_path:
            Path to the input directory, relative to the root directory.
        lightly_path:
            Path to the Lightly directory, relative to the root directory.
        host:
            Host to serve the datasource on.
        port:
            Port to serve the datasource on.

    Examples:
        >>> from lightly.api import serve
        >>> from pathlib import Path
        >>> serve(
        >>>    input_dir=Path("/input_dir),
        >>>    lightly_dir=Path("/lightly_dir),
        >>>    host="localhost",
        >>>    port=1234,
        >>> )

    """
    directories = [root_dir / input_path, root_dir / lightly_path]

    class _LocalDatasourceRequestHandler(SimpleHTTPRequestHandler):
        def translate_path(self, path: str) -> str:
            return _translate_path(path=path, directories=directories)

    return HTTPServer((host, port), _LocalDatasourceRequestHandler)


def _translate_path(path: str, directories: Sequence[Path]) -> str:
    """Translates a relative path to a file in the local datasource.

    Tries to resolve the relative path to a file in the local input directory
    and serves it if it exists. Otherwise, it tries to resolve the relative
    path to a file in the lightly directory and serves it if it exists.

    Args:
        path:
            Relative path to a file in the local datasource.
        directories:
            List of directories to search for the file.


    Returns:
        Absolute path to the file in the local datasource or an empty string
        if the file doesn't exist.

    """
    path = _strip_leading_slashes(path)
    for directory in directories:
        if (directory / path).exists():
            return str(directory / path)
    return ""  # Not found.


def _strip_leading_slashes(path: str) -> str:
    """Strip leading slashes from a path.

    Args:
        path:
            Path to strip leading slashes from.

    Returns:
        Path without leading slashes.

    """
    return re.sub(r"^/+", "", path)
