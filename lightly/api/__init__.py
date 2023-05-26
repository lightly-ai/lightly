""" The lightly.api module provides access to the Lightly API."""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
from lightly.api import patch as _patch
from lightly.api.api_workflow_artifacts import ArtifactNotExist
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import Configuration as _Configuration

# Make ApiWorkflowClient and swagger classes picklable.
_patch.make_swagger_configuration_picklable(
    configuration_cls=_Configuration,
)
