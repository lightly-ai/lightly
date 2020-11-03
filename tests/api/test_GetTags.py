import unittest
import json
import os

import random
import responses

from lightly.api import routes

N = 10


class TestGet(unittest.TestCase):

    def setup(self, psuccess=1.):

        self.psuccess = psuccess

        self.dataset_id = 'XYZ'
        self.token = '123'

        self.tags = {
            'dict': 1, 'of': 2, 'tags': 3
        }

        # set up url
        self.dst_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                                  b'https://api.lightly.ai').decode()
        # route
        self.dst_url += f'/users/datasets/{self.dataset_id}/tags'
        # query
        self.dst_url += f'/?token={self.token}'

    def callback(self, request):
        if random.random() < self.psuccess:
            return (200, [], json.dumps(self.tags))
        else:
            return (500, [], json.dumps({}))

    @responses.activate
    def test_get_tags_all_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=1.)

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        for i in range(N):
            tags = routes.users.datasets.tags.get(self.dataset_id, self.token)
            self.assertDictEqual(tags, self.tags)

    @responses.activate
    def test_get_tags_some_success(self):
        '''Make sure everything works in some error scenario.

        '''

        self.setup(psuccess=.9)

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        for i in range(N):
            tags = routes.users.datasets.tags.get(self.dataset_id, self.token)
            self.assertDictEqual(tags, self.tags)

    @responses.activate
    def test_get_tags_no_success(self):
        '''Make sure everything works in error scenario.

        '''

        self.setup(psuccess=0.)

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        with self.assertRaises(RuntimeError):
            tags = routes.users.datasets.tags.get(self.dataset_id, self.token)
            self.assertDictEqual(tags, self.tags)