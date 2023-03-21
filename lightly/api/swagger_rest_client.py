from typing import Any, Dict, List, Tuple, Union

from lightly.openapi_generated.swagger_client import Configuration
from lightly.openapi_generated.swagger_client.rest import RESTClientObject


class PatchRESTClientObjectMixin:
    """Mixin that adds patches to a RESTClientObject.

    * Adds default timeout to all requests
    * Encodes list query parameters properly
    * Makes the client picklable

    Should only used in combination with RESTClientObject and must come before the
    RESTClientObject in the inheritance order. So this is ok:

        >>> class MyRESTClientObject(PatchRESTClientObjectMixin, RESTClientObject): pass

    while this doesn't work:

        >>> class MyRESTClientObject(RESTClientObject, PatchRESTClientObjectMixin): pass

    A wrong inheritance order will result in the super() calls no calling the correct
    parent classes.
    """

    def __init__(
        self,
        configuration: Configuration,
        timeout: Union[None, int, Tuple[int, int]],
        pools_size: int = 4,
        maxsize: Union[None, int] = None,
    ):
        # Save args as attributes to make the class picklable.
        self.configuration = configuration
        self.timeout = timeout
        self.pools_size = pools_size
        self.maxsize = maxsize

        # Initialize RESTClientObject class
        super().__init__(
            configuration=configuration, pools_size=pools_size, maxsize=maxsize
        )

    def request(
        self,
        method,
        url,
        query_params=None,
        headers=None,
        body=None,
        post_params=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        # Set default timeout. This is necessary because the swagger api client does not
        # respect timeouts configured by urllib3. Instead it expects a timeout to be
        # passed with every request. See code here:
        # https://github.com/lightly-ai/lightly/blob/ffbd32fe82f76b37c8ac497640355314474bfc3b/lightly/openapi_generated/swagger_client/rest.py#L141-L148
        if _request_timeout is None:
            _request_timeout = self.timeout

        flat_query_params = _flatten_list_query_parameters(query_params=query_params)

        # Call RESTClientObject.request
        return super().request(
            method=method,
            url=url,
            query_params=flat_query_params,
            headers=headers,
            body=body,
            post_params=post_params,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
        )

    def __getstate__(self) -> Dict[str, Any]:
        """__getstate__ method for pickling."""
        state = self.__dict__.copy()
        # Delete pool_manager as it cannot be pickled. Note that it is not possible to
        # unpickle and use a LightlySwaggerRESTClientObject without either instantiating
        # the pool_manager manually again or calling the init method on the rest client.
        del state["pool_manager"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """__setstate__ method for pickling."""
        self.__dict__.update(state)
        # Calling init to recreate the pool_manager attribute.
        self.__init__(
            configuration=state["configuration"],
            timeout=state["timeout"],
            pools_size=state["pools_size"],
            maxsize=state["maxsize"],
        )


class LightlySwaggerRESTClientObject(PatchRESTClientObjectMixin, RESTClientObject):
    """Subclass of RESTClientObject which contains additional patches for the request
    method and making the client picklable.

    See PatchRESTClientObjectMixin for details.

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

    pass


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
