import ssl
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
            Hostname for serving the data (defaults to localhost). If you want to expose it to the internet or your local network, use '0.0.0.0'.
            See our docs on lightly-serve for more information: https://docs.lightly.ai/docs/local-storage#view-the-local-data-securely-over-the-networkvpn
        port:
            Port for serving the data (defaults to 3456).
        ssl_key:
            Optional path to the ssl key file.
        ssl_cert:
            Optional path to the ssl cert file.

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

    # setup https/ssl if key or cert are provided
    if cfg.ssl_key or cfg.ssl_cert:
        httpd.socket = ssl.wrap_socket(
            httpd.socket,
            keyfile=Path(cfg.ssl_key) if cfg.ssl_key else None,
            certfile=Path(cfg.ssl_cert) if cfg.ssl_cert else None,
            server_side=True,
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
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


def entry() -> None:
    lightly_serve()
