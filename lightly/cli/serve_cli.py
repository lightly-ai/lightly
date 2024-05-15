import sys
from pathlib import Path

import hydra

from lightly.api import serve
from lightly.api.serve import validate_input_mount, validate_lightly_mount
from lightly.cli._helpers import fix_hydra_arguments
from lightly.utils.hipify import bcolors


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
        print(
            "Please provide a valid 'input_mount' argument. Use --help for more "
            "information."
        )
        sys.exit(1)

    if not cfg.lightly_mount:
        print(
            "Please provide a valid 'lightly_mount' argument. Use --help for more "
            "information."
        )
        sys.exit(1)

    input_mount = Path(cfg.input_mount)
    validate_input_mount(input_mount=input_mount)
    lightly_mount = Path(cfg.lightly_mount)
    validate_lightly_mount(lightly_mount=lightly_mount)

    httpd = serve.get_server(
        paths=[input_mount, lightly_mount],
        host=cfg.host,
        port=cfg.port,
    )
    print(
        f"Starting server, listening at '{bcolors.OKBLUE}{httpd.server_name}:{httpd.server_port}{bcolors.ENDC}'"
    )
    print(
        f"Serving files in '{bcolors.OKBLUE}{cfg.input_mount}{bcolors.ENDC}' and '{bcolors.OKBLUE}{cfg.lightly_mount}{bcolors.ENDC}'"
    )
    print(
        f"Please follow our docs if you are facing any issues: https://docs.lightly.ai/docs/local-storage#optional-after-run-view-local-data-in-lightly-platform"
    )
    httpd.serve_forever()


def entry() -> None:
    lightly_serve()
