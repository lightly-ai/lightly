import sys
from pathlib import Path

import hydra

from lightly.api import serve
from lightly.cli._helpers import fix_hydra_arguments


@hydra.main(**fix_hydra_arguments(config_path="config", config_name="lightly-serve"))
def lightly_serve(cfg):
    """Use lightly-serve to serve your data for interactive exploration.

    Command-Line Args:
        input_dir:
            Path to the input directory.
        lightly_dir:
            Path to the Lightly directory.
        host:
            Hostname for serving the data (defaults to localhost).
        port:
            Port for serving the data.

    Examples:
        >>> lightly-serve input_dir=data/ lightly_dir=lightly/ port=8080


    """
    if not cfg.input_dir:
        print(
            "Please provide a valid input directory. Use --help for more information."
        )
        sys.exit(1)

    if not cfg.lightly_dir:
        print(
            "Please provide a valid Lightly directory. Use --help for more information."
        )
        sys.exit(1)

    if cfg.port is None:
        print("Please provide a valid port. Use --help for more information.")
        sys.exit(1)

    httpd = serve.get_server(
        paths=[Path(cfg.input_dir), Path(cfg.lightly_dir)],
        host=cfg.host,
        port=cfg.port,
    )
    httpd.serve_forever()


def entry() -> None:
    lightly_serve()
