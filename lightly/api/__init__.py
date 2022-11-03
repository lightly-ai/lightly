""" The lightly.api module provides access to the Lightly web-app. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.api.api_workflow_compute_worker import ArtifactNotExist
from lightly.api.patch_rest_client import patch_rest_client

from lightly.openapi_generated.swagger_client.rest import RESTClientObject

# Needed to handle list of arguments correctly
patch_rest_client(RESTClientObject)
