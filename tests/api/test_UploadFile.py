import unittest
import tempfile
import json
import os

import random
import responses

from lightly.api.communication import upload_file_with_signed_url

N = 10


class TestUploadFile(unittest.TestCase):

    def setup(self, psuccess=1.):

        self.psuccess = psuccess
        self.signed_url = 'https://www.this-is-a-signed-url.com'
        self.f = None

    def callback(self, request):
        self.assertEqual(type(request.body), type(self.f))
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
            responses.PUT, self.signed_url,
            callback=self.callback,
            content_type='application/json'
        )

        for _ in range(N):
            self.f = tempfile.NamedTemporaryFile()
            res = upload_file_with_signed_url(self.f, self.signed_url)
            self.assertTrue(res)

    @responses.activate
    def test_put_some_success(self):
        '''Make sure everything works in some error scenario.

        '''

        self.setup(psuccess=.9)

        responses.add_callback(
            responses.PUT, self.signed_url,
            callback=self.callback,
            content_type='application/json'
        )

        for _ in range(N):
            self.f = tempfile.NamedTemporaryFile()
            res = upload_file_with_signed_url(self.f, self.signed_url)
            self.assertTrue(res)

    @responses.activate
    def test_put_no_success(self):
        '''Make sure everything works in some error scenario.

        '''

        self.setup(psuccess=0.)

        responses.add_callback(
            responses.PUT, self.signed_url,
            callback=self.callback,
            content_type='application/json'
        )

        with self.assertRaises(RuntimeError):
            self.f = tempfile.NamedTemporaryFile()
            upload_file_with_signed_url(self.f, self.signed_url)

