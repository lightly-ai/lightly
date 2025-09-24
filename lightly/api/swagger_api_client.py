from typing import Any, Dict, Optional, Tuple, Union

from lightly.api.swagger_rest_client import LightlySwaggerRESTClientObject
from lightly.openapi_generated.swagger_client.api_client import ApiClient, Configuration

DEFAULT_API_TIMEOUT = 60 * 3  # seconds


class PatchApiClientMixin:
    """Mixin that makes an ApiClient object picklable."""

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # Set _pool to None as ThreadPool is not picklable. It will be automatically
        # recreated once the pool is accessed after unpickling.
        state["_pool"] = None
        # Urllib3 response is not picklable. We can safely remove this as it only
        # serves as a cache.
        if "last_response" in state:
            del state["last_response"]
        return state


class LightlySwaggerApiClient(PatchApiClientMixin, ApiClient):
    """Subclass of ApiClient with patches to make the client picklable.

    Uses a LightlySwaggerRESTClientObject instead of RESTClientObject for additional
    patches. See LightlySwaggerRESTClientObject for details.


    Attributes:
        configuration:
            Configuration.
        timeout:
            Timeout in seconds. Is either a single total_timeout value or a
            (connect_timeout, read_timeout) tuple. No timeout is applied if the
            value is None.
            See https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html?highlight=timeout#urllib3.util.Timeout
            for details on the different values.
        header_name:
            A header to pass when making calls to the API.
        header_value:
            A header value to pass when making calls to the API.
        cookie:
            A cookie to include in the header when making calls to the API.
    """

    def __init__(
        self,
        configuration: Configuration,
        timeout: Union[None, int, Tuple[int, int]] = DEFAULT_API_TIMEOUT,
        header_name: Optional[str] = None,
        header_value: Optional[str] = None,
        cookie: Optional[str] = None,
    ):
        super().__init__(
            configuration=configuration,
            header_name=header_name,
            header_value=header_value,
            cookie=cookie,
        )
        self.rest_client = LightlySwaggerRESTClientObject(
            configuration=configuration, timeout=timeout
        )
