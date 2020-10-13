import unittest
import json
import os

import random
import responses

import torchvision
from lightly.api import get_presigned_upload_url


class TestGetPresignedURL(unittest.TestCase):

    def setup(self, n_data=1000, psuccess=1.):

        self.filename = 'filename'
        self.dataset_id = 'XYZ'
        self.sample_id = 'ABC'
        self.token = 'secret'

        self.psuccess = psuccess

        # set up url
        self.dst_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                                  b'https://api.whattolabel.com').decode()
        # route
        self.dst_url += f'/users/datasets/{self.dataset_id}'
        self.dst_url += f'/samples/{self.sample_id}/writeurl'
        # query
        self.dst_url += f'?fileName={self.filename}&token={self.token}'

        # create a dataset
        self.dataset = torchvision.datasets.FakeData(size=n_data,
                                                     image_size=(3, 32, 32))

    def signed_url_callback(self, request):
        resp_body = {'signedWriteUrl': 'https://this-is-a-signed-url.com'}
        if random.random() < self.psuccess:
            return (200, [], json.dumps(resp_body))
        else:
            return (500, [], json.dumps({}))

    @responses.activate
    def test_signed_url_all_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(n_data=100, psuccess=1.)

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.signed_url_callback,
            content_type='application/json'
        )

        for f in self.dataset:
            url = get_presigned_upload_url(
                'filename',
                self.dataset_id,
                self.sample_id,
                self.token
            )
            self.assertEqual(url, 'https://this-is-a-signed-url.com')

    @responses.activate
    def test_signed_url_some_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(n_data=100, psuccess=.9)

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.signed_url_callback,
            content_type='application/json'
        )

        for f in self.dataset:
            url = get_presigned_upload_url(
                'filename',
                self.dataset_id,
                self.sample_id,
                self.token
            )
            self.assertEqual(url, 'https://this-is-a-signed-url.com')

    @responses.activate
    def test_signed_url_no_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(n_data=1, psuccess=.0)

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.signed_url_callback,
            content_type='application/json'
        )

        with self.assertRaises(RuntimeError):
            for f in self.dataset:
                url = get_presigned_upload_url(
                    'filename',
                    self.dataset_id,
                    self.sample_id,
                    self.token
                )
                self.assertEqual(url, 'https://this-is-a-signed-url.com')
