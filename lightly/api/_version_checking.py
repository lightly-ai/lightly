from threading import Thread

from lightly.api import utils
from lightly.api.swagger_api_client import LightlySwaggerApiClient
from lightly.openapi_generated.swagger_client.api import VersioningApi
from lightly.utils import version_compare

# Default timeout for API version verification requests in seconds.
DEFAULT_TIMEOUT_SEC = 2


def is_latest_version(current_version: str) -> bool:
    """Returns True if package version is latest released version."""
    latest_version = get_latest_version(current_version)
    return version_compare.version_compare(current_version, latest_version) >= 0


def is_compatible_version(current_version: str) -> bool:
    """Returns True if package version is compatible with API."""
    minimum_version = get_minimum_compatible_version()
    return version_compare.version_compare(current_version, minimum_version) >= 0


def get_latest_version(
    current_version: str, timeout_sec: float = DEFAULT_TIMEOUT_SEC
) -> str:
    """Returns the latest package version."""
    versioning_api = _get_versioning_api()
    version_number: str = versioning_api.get_latest_pip_version(
        current_version=current_version,
        _request_timeout=timeout_sec,
    )
    return version_number


def get_minimum_compatible_version(
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> str:
    """Returns minimum package version that is compatible with the API."""
    versioning_api = _get_versioning_api()
    version_number: str = versioning_api.get_minimum_compatible_pip_version(
        _request_timeout=timeout_sec
    )
    return version_number


def check_is_latest_version_in_background(current_version: str) -> None:
    """Checks if the current version is the latest version in a background thread."""

    def _check_version_in_background(current_version: str) -> None:
        try:
            is_latest_version(current_version=current_version)
        except Exception:
            # Ignore failed check.
            pass

    thread = Thread(
        target=_check_version_in_background,
        kwargs=dict(current_version=current_version),
        daemon=True,
    )
    thread.start()


def _get_versioning_api() -> VersioningApi:
    configuration = utils.get_api_client_configuration(
        raise_if_no_token_specified=False,
    )
    # Set retries to 0 to avoid waiting for retries in case of a timeout.
    configuration.retries = 0
    api_client = LightlySwaggerApiClient(configuration=configuration)
    versioning_api = VersioningApi(api_client=api_client)
    return versioning_api
