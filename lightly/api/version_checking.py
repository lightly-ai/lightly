import signal
import warnings
from multiprocessing import current_process
from typing import Tuple

from lightly.openapi_generated.swagger_client import VersioningApi
from lightly.openapi_generated.swagger_client.api_client import ApiClient

import lightly.api.api_workflow_client as api_workflow_client
from lightly.api.utils import getenv
from lightly.utils import version_compare


class LightlyAPITimeoutException(Exception):
    pass

class TimeoutDecorator:
    def __init__(self, seconds):
        self.seconds = seconds

    def handle_timeout_method(self, *args, **kwargs):
        raise LightlyAPITimeoutException

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout_method)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)


def do_version_check(current_version: str):
    if current_process().name == 'MainProcess':
        with TimeoutDecorator(1):
            versioning_api = get_versioning_api()
            current_version: str = versioning_api.get_latest_pip_version(
                current_version=current_version)
            latest_version: str = versioning_api.get_minimum_compatible_pip_version()

            try:
                if version_compare(current_version, latest_version) < 0:
                    # local version is behind latest version
                    pretty_print_latest_version(current_version, latest_version)
            except ValueError:
                pass
                # error during version compare
                # we just skip the version check



def get_versioning_api() -> VersioningApi:
    configuration = api_workflow_client.get_api_client_configuration_no_token()
    configuration.host = getenv('LIGHTLY_SERVER_LOCATION', 'https://api.lightly.ai')
    api_client = ApiClient(configuration=configuration)
    versioning_api = VersioningApi(api_client)
    return versioning_api


def get_latest_version(current_version: str) -> Tuple[None, str]:
    try:
        versioning_api = get_versioning_api()
        version_number: str = versioning_api.get_latest_pip_version(current_version=current_version)
        return version_number
    except Exception as e:
        return None


def get_minimum_compatible_version():
    versioning_api = get_versioning_api()
    version_number: str = versioning_api.get_minimum_compatible_pip_version()
    return version_number


def pretty_print_latest_version(current_version, latest_version, width=70):
    warning = f"You are using lightly version {current_version}. " \
              f"There is a newer version of the package available. " \
              f"For compatability reasons, please upgrade your current version: " \
              f"pip install lightly=={latest_version}"
    warnings.warn(Warning(warning))
