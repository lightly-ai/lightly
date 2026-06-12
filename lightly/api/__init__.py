"""Deprecated compatibility module for the removed Lightly API client."""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

_ERROR_MESSAGE = (
    "lightly.api and ApiWorkflowClient have been removed from the lightly package. "
    "The API workflow client is deprecated. If you need ApiWorkflowClient, use "
    "Lightly SSL version v1.15.x or older, for example with "
    '`pip install "lightly<1.16"`.'
)

raise ImportError(_ERROR_MESSAGE)
