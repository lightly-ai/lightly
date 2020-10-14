import unittest
import json
import os

import random
import responses

from lightly.api.communication import create_initial_tag

N = 10


class TestCreateInitialTag(unittest.TestCase):

    def setup(self, psuccess=1.):

        self.psuccess = psuccess

        self.dataset_id = 'XYZ'
        self.token = '123'

        # set up url
        self.dst_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                                  b'https://api.lightly.ai').decode()
        # route
        self.dst_url += f'/users/datasets/{self.dataset_id}/tags'

    def callback(self, request):
        body = json.loads(request.__dict__['body'].decode())
        self.assertEqual(body['token'], self.token)
        if random.random() < self.psuccess:
            return (200, [], json.dumps({}))
        else:
            return (500, [], json.dumps({}))

    @responses.activate
    def test_create_initial_tag_all_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=1.)

        responses.add_callback(
            responses.POST, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        for i in range(N):
            create_initial_tag(self.dataset_id, self.token)

    @responses.activate
    def test_create_initial_tag_some_success(self):
        '''Make sure everything works in some error scenario.

        '''

        self.setup(psuccess=.9)

        responses.add_callback(
            responses.POST, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        for i in range(N):
            create_initial_tag(self.dataset_id, self.token)

    @responses.activate
    def test_create_initial_tag_no_success(self):
        '''Make sure everything works in error scenario.

        '''

        self.setup(psuccess=0.)

        responses.add_callback(
            responses.POST, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        with self.assertRaises(RuntimeError):
            create_initial_tag(self.dataset_id, self.token)
