import unittest
import json
import os

import time
import random
import responses

import lightly
from lightly.api import get_latest_version

N = 10


class TestGetLatestVersion(unittest.TestCase):

    def setup(self, psuccess=1.):

        self.psuccess = psuccess

        # set up url
        self.dst_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                                  b'https://api.lightly.ai').decode()
        # route
        self.dst_url += '/pip/version'
        self.version = '0.0.0'

    def callback(self, request):
        if random.random() < self.psuccess:
            return (200, [], json.dumps([self.version]))
        else:
            return (500, [], json.dumps({}))

    def timeout_callback(self, request):
        time.sleep(1)
        return (200, [], json.dumps([self.version]))


    @responses.activate
    def test_get_latest_version_all_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=1.)

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        t0 = time.time()
        version = get_latest_version(self.version)
        t1 = time.time()

        self.assertLessEqual(t1 - t0, 1)
        self.assertEqual(version, self.version)

    @responses.activate
    def test_get_latest_version_no_success(self):
        '''Make sure everything works in error scenario.

        '''

        self.setup(psuccess=0.)

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.callback,
            content_type='application/json'
        )

        t0 = time.time()
        version = get_latest_version(self.version)
        t1 = time.time()

        self.assertLessEqual(t1 - t0, 1)
        self.assertIsNone(version)

    @responses.activate
    def test_get_latest_version_timeout(self):
        '''Make sure everything works in error scenario.

        '''

        self.setup(psuccess=1.)

        responses.add_callback(
            responses.GET, self.dst_url,
            callback=self.timeout_callback,
            content_type='application/json'
        )

        t0 = time.time()
        version = get_latest_version(self.version)
        t1 = time.time()

        self.assertLessEqual(t1 - t0, 1.05)
        self.assertEqual(version, self.version)
