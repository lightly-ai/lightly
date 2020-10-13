import unittest
import json
import os

import random
import responses

from lightly.api.communication import _get_request

N = 10


class TestGet(unittest.TestCase):

    def setup(self, psuccess=1.):

        self.psuccess = psuccess

        # set up url
        self.dst_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                                  b'https://api.whattolabel.com').decode()
        # route
        self.dst_url += '/sample/route/to/get'
        # query
        self.dst_url += '?sample=query'

    def callback(self, request):
        if random.random() < self.psuccess:
            return (200, [], json.dumps({}))
        else:
            return (500, [], json.dumps({}))

    @responses.activate
    def test_get_all_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=1.)

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        for i in range(N):
            params = {'sample': 'query'}
            _get_request(self.dst_url, params=params)

    @responses.activate
    def test_get_some_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=.9)

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        for i in range(N):
            params = {'sample': 'query'}
            _get_request(self.dst_url, params=params)

    @responses.activate
    def test_get_no_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=0.)

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        with self.assertRaises(RuntimeError):
            params = {'sample': 'query'}
            _get_request(self.dst_url, params=params)
