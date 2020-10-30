import unittest
import json
import os

import random
import responses

import lightly.api.routes as routes

N = 10


class TestGet(unittest.TestCase):

    def setup(self, mode='tag_exists', psuccess=1.):

        self.mode = mode
        self.psuccess = psuccess

        self.dataset_id = 'XYZ'
        self.token = '123'
        
        self.tag_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                                  b'https://api.lightly.ai').decode()
        self.tag_url += f'/users/datasets/{self.dataset_id}/tags/'
        self.tags = [
            {'name': 'initial-tag', '_id': '123'},
            {'name': 'test-tag', '_id': '456'},
        ]

        self.dst_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                                  b'https://api.lightly.ai').decode()
        self.dst_url += f'/users/datasets/{self.dataset_id}/tags/123/download'
        self.samples = 'sample_1.jpg\nsample_2.jpg'

    def tags_callback(self, request):
        if random.random() < self.psuccess:
            if self.mode == 'tag_exists':
                return (200, [], json.dumps(self.tags))
            else:
                return (200, [], json.dumps([]))
        else:
            return (500, [], json.dumps({}))

    def samples_callback(self, request):
        if random.random() < self.psuccess:
            return (200, [], self.samples)
        else:
            return (500, [], json.dumps({}))


    @responses.activate
    def test_get_samples_all_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=1.)

        responses.add_callback(
            responses.GET, self.tag_url,
            callback=self.tags_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.samples_callback,
            content_type='application/json'
        )

        for i in range(N):
            samples = routes.users.datasets.tags.get_samples(self.dataset_id, self.token)
            for s0, s1 in zip(samples, self.samples.splitlines()):
                self.assertEqual(s0, s1)

    @responses.activate
    def test_get_samples_some_success(self):
        '''Make sure everything works in some error scenario.

        '''

        self.setup(psuccess=.9)

        responses.add_callback(
            responses.GET, self.tag_url,
            callback=self.tags_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.samples_callback,
            content_type='application/json'
        )

        for i in range(N):
            samples = routes.users.datasets.tags.get_samples(self.dataset_id, self.token)
            for s0, s1 in zip(samples, self.samples.splitlines()):
                self.assertEqual(s0, s1)
    
    @responses.activate
    def test_get_samples_no_success(self):
        '''Make sure everything works in error scenario.

        '''

        self.setup(psuccess=0.)

        responses.add_callback(
            responses.GET, self.tag_url,
            callback=self.tags_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.samples_callback,
            content_type='application/json'
        )

        with self.assertRaises(RuntimeError):
            samples = routes.users.datasets.tags.get_samples(self.dataset_id, self.token)
            for s0, s1 in zip(samples, self.samples.splitlines()):
                self.assertEqual(s0, s1)

    @responses.activate
    def test_get_samples_no_tag(self):
        '''Make sure everything works in error scenario.

        '''

        self.setup(psuccess=1., mode='no_tag_exists')

        responses.add_callback(
            responses.GET, self.tag_url,
            callback=self.tags_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.samples_callback,
            content_type='application/json'
        )

        with self.assertRaises(RuntimeError):
            samples = routes.users.datasets.tags.get_samples(self.dataset_id, self.token)
            for s0, s1 in zip(samples, self.samples.splitlines()):
                self.assertEqual(s0, s1)
