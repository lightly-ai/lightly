from typing import Any, Dict, Tuple, Union

from lightly.api.swagger_rest_client import LightlySwaggerRESTClientObject
from lightly.openapi_generated.swagger_client import ApiClient

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
        pools_size:
            Number of connection pools. Defaults to 4.
        maxsize:
            Maxsize is the number of requests to host that are allowed in parallel.
            Defaults to None.
    """

    def __init__(
        self,
        configuration,
        timeout: Union[None, int, Tuple[int, int]] = DEFAULT_API_TIMEOUT,
        header_name=None,
        header_value=None,
        cookie=None,
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
