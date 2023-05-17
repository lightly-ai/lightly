from typing import Any, Dict, List, Tuple, Union

from lightly.api.swagger_rest_client import LightlySwaggerRESTClientObject
from lightly.openapi_client.api_client import ApiClient, Configuration

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

    def call_api(
        self,
        resource_path,
        method,
        path_params=None,
        query_params=None,
        header_params=None,
        body=None,
        post_params=None,
        files=None,
        response_types_map=None,
        auth_settings=None,
        async_req=None,
        _return_http_data_only=None,
        collection_formats=None,
        _preload_content=True,
        _request_timeout=None,
        _host=None,
        _request_auth=None,
    ):
        query_params = _flatten_list_query_parameters(query_params)
        return super().call_api(
            resource_path,
            method,
            path_params=path_params,
            query_params=query_params,
            header_params=header_params,
            body=body,
            post_params=post_params,
            files=files,
            response_types_map=response_types_map,
            auth_settings=auth_settings,
            async_req=async_req,
            _return_http_data_only=_return_http_data_only,
            collection_formats=collection_formats,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            _host=_host,
            _request_auth=_request_auth,
        )


def _flatten_list_query_parameters(
    query_params: Union[None, List[Tuple[str, Any]]]
) -> Union[None, List[Tuple[str, Any]]]:
    if query_params is not None:
        new_query_params = []
        for name, value in query_params:
            if isinstance(value, list):
                new_query_params.extend([(name, val) for val in value])
            else:
                new_query_params.append((name, value))
        query_params = new_query_params
    return query_params


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
        header_name: Union[str, None] = None,
        header_value: Union[str, None] = None,
        cookie: Union[str, None] = None,
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
