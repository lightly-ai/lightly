import unittest
import json
import os

import random
import responses
import pytest


from lightly.api.utils import put_request

N = 10

@pytest.mark.slow
class TestPut(unittest.TestCase):

    def setup(self, psuccess=1.):

        self.psuccess = psuccess

        # set up url
        self.dst_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                                  b'https://api.lightly.ai').decode()
        # route
        self.dst_url += '/sample/route/to/put'

    def callback(self, request):
        self.assertEqual('sample=data', request.body)
        if random.random() < self.psuccess:
            return (200, [], json.dumps({}))
        else:
            return (500, [], json.dumps({}))

    @responses.activate
    def test_put_all_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=1.)

        responses.add_callback(
            responses.PUT, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        for _ in range(N):
            data = {'sample': 'data'}
            put_request(self.dst_url, data)

    @responses.activate
    def test_put_some_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=.9)

        responses.add_callback(
            responses.PUT, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        for _ in range(N):
            data = {'sample': 'data'}
            put_request(self.dst_url, data)

    @responses.activate
    def test_put_no_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=0.)

        responses.add_callback(
            responses.PUT, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        with self.assertRaises(RuntimeError):
            data = {'sample': 'data'}
            put_request(self.dst_url, data)