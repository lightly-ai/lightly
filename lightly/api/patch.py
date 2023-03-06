import logging
from typing import Any, Dict, Type


def rest_client_flatten_array_query_parameters(rest_client: Type):
    """

    Patches the rest client to flatten out array query parameters.

    Example:
        query_params: [('labels', ['AAA', 'BBB'])]
        new_query_params: [('labels', 'AAA'), ('labels', 'BBB')]
        url part in query: "labels=AAA&labels=BBB"

        Without this patch, the query_params would be translated into
        "labels=['AAA', 'BBB']" in the url itself, which fails.

    Args:
        rest_client:
            Must be the class swagger_client.rest.RESTClientObject to patch.
            Note: it must be the class itself, not an instance of it.

    Returns:

    """
    request = rest_client.request

    def request_patched(
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
        if query_params is not None:
            new_query_params = []
            for name, value in query_params:
                if isinstance(value, list):
                    new_query_params.extend([(name, val) for val in value])
                else:
                    new_query_params.append((name, value))
            query_params = new_query_params
        return request(
            method=method,
            url=url,
            query_params=query_params,
            headers=headers,
            body=body,
            post_params=post_params,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
        )

    return request_patched


def make_swagger_generated_classes_picklable(
    api_client_cls: Type,
    configuration_cls: Type,
    rest_client_cls: Type,
) -> None:
    """Adds __getstate__ and __setstate__ methods to swagger generated classes to make
    them picklable."""
    api_client_cls.__setstate__ = _ApiClient__setstate__
    api_client_cls.__getstate__ = _ApiClient__getstate__
    configuration_cls.__getstate__ = _Configuration__getstate__
    configuration_cls.__setstate__ = _Configuration__setstate__
    rest_client_cls.__getstate__ = _RESTClientObject__getstate__


def _Configuration__getstate__(self) -> Dict[str, Any]:
    state = self.__dict__.copy()
    # Remove unpicklable entries.
    state["logger"] = {}
    state["logger_formatter"] = None
    state["logger_stream_handler"] = None
    state["logger_file_handler"] = None
    return state


def _Configuration__setstate__(self, state: Dict[str, Any]) -> None:
    self.__dict__.update(state)
    # Recreate logger objects.
    self.logger["package_logger"] = logging.getLogger(
        "lightly.openapi_generated.swagger_client"
    )
    self.logger["urllib3_logger"] = logging.getLogger("urllib3")
    # Set logger_format and logger_file explicitly because they recreate
    # logger_formatter, logger_file_handler, and logger_stream_handler which are removed
    # before pickling.
    self.logger_format = state["_Configuration__logger_format"]
    self.logger_file = state["_Configuration__logger_file"]
    # Set debug explicitly because it has side effects.
    self.debug = state["_Configuration__debug"]


def _RESTClientObject__getstate__(self) -> Dict[str, Any]:
    state = self.__dict__.copy()
    # Delete pool_manager as it cannot be pickled.
    # Note that it is not possible to unpickle and use a RESTClientObject without
    # either instantiating the pool_manager manually again or calling the init method
    # on the rest client. However, calling init requires a configuration object which
    # is not saved on the RESTClientObject and can only be passed from outside.
    del state["pool_manager"]
    return state


def _ApiClient__getstate__(self) -> Dict[str, Any]:
    state = self.__dict__.copy()
    # ThreadPool is not picklable. We set it to None as it will be automatically
    # recreated once the pool is accessed after unpickling.
    state["_pool"] = None
    # Urllib3 response is not picklable. We can safely remove this as it only serves as
    # a cache.
    del state["last_response"]
    return state


def _ApiClient__setstate__(self, state: Dict[str, Any]) -> None:
    self.__dict__.update(state)
    # We have to call init on rest_client to fully instantiate the rest client again
    # and recreate the pool manager which is removed before pickling.
    # See _RESTClientObject__getstate__ for details.
    self.rest_client.__init__(state["configuration"])
