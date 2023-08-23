import sys
from pathlib import Path

import hydra

from lightly.api import serve
from lightly.cli._helpers import fix_hydra_arguments


@hydra.main(**fix_hydra_arguments(config_path="config", config_name="lightly-serve"))
def lightly_serve(cfg):
    """Use lightly-serve to serve your data for interactive exploration.

    Command-Line Args:
        input_mount:
            Path to the input directory.
        lightly_mount:
            Path to the Lightly directory.
        host:
            Hostname for serving the data (defaults to localhost).
        port:
            Port for serving the data (defaults to 3456).

    Examples:
        >>> lightly-serve input_mount=data/ lightly_mount=lightly/ port=3456


    """
    if not cfg.input_mount:
        print("Please provide a valid input mount. Use --help for more information.")
        sys.exit(1)

    if not cfg.lightly_mount:
        print("Please provide a valid Lightly mount. Use --help for more information.")
        sys.exit(1)

    httpd = serve.get_server(
        paths=[Path(cfg.input_mount), Path(cfg.lightly_mount)],
        host=cfg.host,
        port=cfg.port,
    )
    print(f"Starting server, listening at '{httpd.server_name}:{httpd.server_port}'")
    print(f"Serving files in '{cfg.input_mount}' and '{cfg.lightly_mount}'")
    httpd.serve_forever()


def entry() -> None:
    lightly_serve()
