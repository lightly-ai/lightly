""" The lightly.api module provides access to the Lightly web-app. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
from lightly.api import patch as _patch
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.api.api_workflow_compute_worker import ArtifactNotExist

from lightly.openapi_generated.swagger_client import ApiClient as _ApiClient, Configuration as _Configuration
from lightly.openapi_generated.swagger_client.rest import RESTClientObject as _RESTClientObject

# Handle requests with list of query parameters correctly.
_patch.rest_client_flatten_array_query_parameters(_RESTClientObject)

# Make ApiWorkflowClient and swagger classes picklable.
_patch.make_swagger_generated_classes_picklable(
    api_client_cls=_ApiClient,
    configuration_cls=_Configuration,
    rest_client_cls=_RESTClientObject,
)
