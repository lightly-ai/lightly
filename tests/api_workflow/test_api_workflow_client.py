import platform
import unittest
from unittest import mock

import pytest
import requests
import os

from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient
from lightly.api.api_workflow_client import LIGHTLY_S3_SSE_KMS_KEY
from lightly.__init__ import __version__

class TestApiWorkflowClient(unittest.TestCase):

    def test_upload_file_with_signed_url(self):
        with mock.patch('lightly.api.api_workflow_client.requests') as requests:
            client = ApiWorkflowClient(token="")
            file = mock.Mock()
            signed_write_url = ''
            client.upload_file_with_signed_url(
                file=file,
                signed_write_url=signed_write_url,
            )
            requests.put.assert_called_with(signed_write_url, data=file)

    def test_upload_file_with_signed_url_session(self):
        session = mock.Mock()
        file = mock.Mock()
        signed_write_url = ''
        client = ApiWorkflowClient(token="")
        client.upload_file_with_signed_url(
            file=file,
            signed_write_url=signed_write_url,
            session=session
        )
        session.put.assert_called_with(signed_write_url, data=file)
    
    def test_upload_file_with_signed_url_session_sse(self):
        session = mock.Mock()
        file = mock.Mock()
        signed_write_url = 'http://somwhere.s3.amazonaws.com/someimage.png'
        client = ApiWorkflowClient(token="")
        # set the environment var to enable SSE 
        os.environ[LIGHTLY_S3_SSE_KMS_KEY] = 'True'
        client.upload_file_with_signed_url(
            file=file,
            signed_write_url=signed_write_url,
            session=session
        )
        session.put.assert_called_with(signed_write_url, data=file, headers={'x-amz-server-side-encryption': 'AES256'})
    
    def test_upload_file_with_signed_url_session_sse_kms(self):
        session = mock.Mock()
        file = mock.Mock()
        signed_write_url = 'http://somwhere.s3.amazonaws.com/someimage.png'
        client = ApiWorkflowClient(token="")
        # set the environment var to enable SSE with KMS 
        sseKMSKey = "arn:aws:kms:us-west-2:123456789000:key/1234abcd-12ab-34cd-56ef-1234567890ab"
        os.environ[LIGHTLY_S3_SSE_KMS_KEY] = sseKMSKey
        client.upload_file_with_signed_url(
            file=file,
            signed_write_url=signed_write_url,
            session=session
        )
        session.put.assert_called_with(
            signed_write_url,
            data=file,
            headers={
                'x-amz-server-side-encryption': 'aws:kms',
                'x-amz-server-side-encryption-aws-kms-key-id': sseKMSKey,
            }
        )

    def test_upload_file_with_signed_url_raise_status(self):
        def raise_connection_error(*args, **kwargs):
            raise requests.exceptions.ConnectionError()

        with mock.patch('lightly.api.api_workflow_client.requests.put', raise_connection_error):
            client = ApiWorkflowClient(token="")
            with self.assertRaises(requests.exceptions.ConnectionError):
                client.upload_file_with_signed_url(
                    file=mock.Mock(),
                    signed_write_url='',
                )

def test_user_agent(mocker: MockerFixture) -> None:
    client = ApiWorkflowClient(token="")
    patched_request = mocker.patch.object(client._mappings_api.api_client, "request")

    # We use get_sample_mappings_by_dataset_id as one test method to ensure that
    # it includes the user agent in the header.
    # After calling the patched request, we don't care about the RESTResponse
    # and mocking it properly is complicated,
    # thus we just stop API call after the patched request function is called.
    with pytest.raises(TypeError, match="the JSON object must be str, bytes or bytearray, not MagicMock"):
        client._mappings_api.get_sample_mappings_by_dataset_id(dataset_id="")

    assert patched_request.call_args.kwargs["headers"]["User-Agent"] == f"Lightly/{__version__}/python ({platform.platform()})"
