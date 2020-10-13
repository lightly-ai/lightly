import unittest
import json
import os

import random
import responses

from lightly.api.communication import _post_request

N = 10


class TestPost(unittest.TestCase):

    def setup(self, psuccess=1.):

        self.psuccess = psuccess

        # set up url
        self.dst_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                                  b'https://api.whattolabel.com').decode()
        # route
        self.dst_url += '/sample/route/to/post'

        self.data = {'sample': 'data'}
        self.json = {'sample': 'json'}

    def callback(self, request):
        self.assertEqual('sample=data', request.body)
        if random.random() < self.psuccess:
            return (200, [], json.dumps({}))
        else:
            return (500, [], json.dumps({}))

    @responses.activate
    def test_post_all_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=1.)

        responses.add_callback(
            responses.POST, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        for i in range(N):
            data = self.data
            json = self.json
            _post_request(self.dst_url, data=data, json=json)

    @responses.activate
    def test_post_some_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=.9)

        responses.add_callback(
            responses.POST, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        for i in range(N):
            data = self.data
            json = self.json
            _post_request(self.dst_url, data=data, json=json)

    @responses.activate
    def test_post_no_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=0.)

        responses.add_callback(
            responses.POST, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        with self.assertRaises(RuntimeError):
            data = self.data
            json = self.json
            _post_request(self.dst_url, data=data, json=json)
