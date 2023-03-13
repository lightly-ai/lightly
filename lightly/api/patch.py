import logging
from typing import Any, Dict, Type


def make_swagger_configuration_picklable(
    configuration_cls: Type,
) -> None:
    """Adds __getstate__ and __setstate__ methods to swagger configuration to make it
    picklable.

    This doesn't make all swagger classes picklable. Notably, the ApiClient and
    and RESTClientObject classes are not picklable. Use the picklable
    LightlySwaggerApiClient and LightlySwaggerRESTClientObject classes instead.
    """
    configuration_cls.__getstate__ = _Configuration__getstate__
    configuration_cls.__setstate__ = _Configuration__setstate__


def _Configuration__getstate__(self: Type) -> Dict[str, Any]:
    state = self.__dict__.copy()

    # Remove all loggers as they are not picklable. This removes the package_logger
    # and urllib3_logger. Note that we cannot remove the with:
    # `del state["logger"]["package_logger"]` as this would modify self.__dict__ due to
    # shallow copy.
    state["logger"] = {}

    # formatter, stream_handler and file_handler are not picklable
    state["logger_formatter"] = None
    state["logger_stream_handler"] = None
    state["logger_file_handler"] = None
    return state


def _Configuration__setstate__(self: Type, state: Dict[str, Any]) -> None:
    self.__dict__.update(state)
    # Recreate logger objects.
    self.logger["package_logger"] = logging.getLogger(
        "lightly.openapi_generated.swagger_client"
    )
    self.logger["urllib3_logger"] = logging.getLogger("urllib3")

    # Set logger_format and logger_file explicitly because they have setter decoraters
    # defined on the Configuration class. These decorates have side effects and create
    # self.__logger_format, self.logger_formatter, self.__logger_file,
    # self.logger_file_handler, and self.logger_stream_handler under the hood.
    #
    # Note that the setter decorates are not called by the self.__dict__.update(state)
    # at the beginning of the function.
    #
    # The attributes are set to the class mangled values stored in the state dict.
    self.logger_format = state[
        "_Configuration__logger_format"
    ]  # set to self.__logger_format
    self.logger_file = state["_Configuration__logger_file"]  # set to self.__logger_file

    # Set debug explicitly because it has a setter decorator with side effects.
    self.debug = state["_Configuration__debug"]
