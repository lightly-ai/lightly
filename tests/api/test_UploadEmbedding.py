import unittest
import json
import os

import random
import responses

from lightly.api.communication import upload_embedding

N = 10


class TestUploadSample(unittest.TestCase):

    def setup(self, psuccess=1.):

        self.psuccess = psuccess

        self.dataset_id = 'XYZ'
        self.token = '123'

        self.data_1 = {
            'embeddingName': 'default',
            'embeddings': [0.1 * i for i in range(16)],
            'token': '123',
            'datasetId': 'XYZ',
            'append': 0
        }

        self.dst_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                                  b'https://api.lightly.ai').decode()
        self.dst_url += f'/users/datasets/{self.dataset_id}/embeddings'

    def callback_1(self, request):
        body = json.loads(request.body.decode())
        self.assertDictContainsSubset(body, self.data_1)
        if random.random() < self.psuccess:
            return (200, [], json.dumps({'sampleId': 'sample_id'}))
        else:
            return (500, [], json.dumps({}))

    @responses.activate
    def test_upload_embedding_all_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=1.)

        responses.add_callback(
            responses.POST, self.dst_url,
            callback=self.callback_1,
            content_type='application/json'
        )

        for i in range(N):
            upload_embedding(self.data_1)

    @responses.activate
    def test_upload_embedding_some_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=.9)

        responses.add_callback(
            responses.POST, self.dst_url,
            callback=self.callback_1,
            content_type='application/json'
        )

        for i in range(N):
            upload_embedding(self.data_1)

    @responses.activate
    def test_upload_embedding_no_success(self):
        '''Make sure everything works in no-error scenario.

        '''

        self.setup(psuccess=0.)

        responses.add_callback(
            responses.POST, self.dst_url,
            callback=self.callback_1,
            content_type='application/json'
        )

        with self.assertRaises(RuntimeError):
            upload_embedding(self.data_1)
