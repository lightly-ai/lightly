import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def serve(
    root_dir: Path,
    input_path: str,
    lightly_path: str,
    host: str,
    port: int,
):
    """Serve the local datasource.

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
        """Request handler for the local datasource.

        Tries to resolve the relative path to a file in the local input directory
        and serves it if it exists. Otherwise, it tries to resolve the relative
        path to a file in the lightly directory and serves it if it exists.

        """

        def translate_path(self, path: str) -> str:
            path = _strip_leading_slashes(path)
            for directory in directories:
                if (directory / path).is_file():
                    return str(directory / path)
            return ""  # Not found.

    httpd = HTTPServer((host, port), _LocalDatasourceRequestHandler)
    httpd.serve_forever()


def _strip_leading_slashes(path: str) -> str:
    """Strip leading slashes from a path.

    Args:
        path:
            Path to strip leading slashes from.

    Returns:
        Path without leading slashes.

    """
    return re.sub(r"^/+", "", path)
