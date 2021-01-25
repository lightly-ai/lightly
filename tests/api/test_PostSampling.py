import unittest
import tempfile
import json
import os

import random
import responses
import pytest


from lightly.api.active_learning import sampling_request_to_api
from lightly.openapi_generated.swagger_client import SamplingMethod


@pytest.mark.slow
class TestPostSampling(unittest.TestCase):

    def setup(self, psuccess=1.):

        self.psuccess = psuccess
        self.signed_url = 'https://www.this-is-a-signed-url.com'

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
            responses.POST, self.signed_url,
            callback=self.callback,
            content_type='application/json'
        )

        token = 'bb10724138a5b33a0f35c444'
        dataset_id = '6006f54aab0cd9000ad7914c'
        tag_id = 'initial-tag'
        embedding_id = '0'
        sampling_name = 'sampling-test'
        sampling_method = SamplingMethod.RANDOM
        n_samples = 100
        min_distance = 0.1
        response_job_id = sampling_request_to_api(token, dataset_id, tag_id, embedding_id,
                                                  sampling_name, sampling_method, n_samples, min_distance)


