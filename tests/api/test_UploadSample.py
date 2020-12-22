import unittest
import tempfile
import json
import os

import random
import responses
import pytest

from lightly.api import routes

N = 10

@pytest.mark.slow
class TestUploadSample(unittest.TestCase):

    def setup(self, psuccess=1.):

        self.psuccess = psuccess

        self.dataset_id = 'XYZ'
        self.token = '123'

        self.filename = 'filename'
        self.thumbname = 'thumbname'
        self.metadata = {
            'example': 0.1,
            'metadata': 'this'
        }

        self.dst_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                                  b'https://api.lightly.ai').decode()
        self.dst_url += f'/users/datasets/{self.dataset_id}/samples/'

    def callback(self, request):
        body = json.loads(request.body.decode())['sample']
        self.assertEqual(body['fileName'], self.filename)
        self.assertEqual(body['thumbName'], self.thumbname)
        self.assertDictEqual(body['meta'], self.metadata)
        if random.random() < self.psuccess:
            return (200, [], json.dumps({'sampleId': 'sample_id'}))
        else:
            return (500, [], json.dumps({}))

    @responses.activate
    def test_upload_sample_all_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=1.)

        responses.add_callback(
            responses.POST, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        for _ in range(N):
            self.f = tempfile.NamedTemporaryFile()
            routes.users.datasets.samples.post(
                self.filename,
                self.thumbname,
                self.metadata,
                self.dataset_id,
                self.token
            )

    @responses.activate
    def test_upload_sample_some_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=.9)

        responses.add_callback(
            responses.POST, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        for _ in range(N):
            self.f = tempfile.NamedTemporaryFile()
            routes.users.datasets.samples.post(
                self.filename,
                self.thumbname,
                self.metadata,
                self.dataset_id,
                self.token
            )

    @responses.activate
    def test_upload_sample_no_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=0.)

        responses.add_callback(
            responses.POST, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        with self.assertRaises(RuntimeError):
            self.f = tempfile.NamedTemporaryFile()
            routes.users.datasets.samples.post(
                self.filename,
                self.thumbname,
                self.metadata,
                self.dataset_id,
                self.token
            )
