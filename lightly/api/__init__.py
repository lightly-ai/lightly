""" The lightly.api module provides access to the Lightly web-app. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.api.api_workflow_compute_worker import ArtifactNotExist

# Mock the urlencode to use doseq, so that arrays in the query are properly encoded
from six.moves.urllib.parse import urlencode
def doseq_urlencode(fields):
    return urlencode(fields, doseq=True)
import urllib3.request
urllib3.request.urlencode = doseq_urlencode
