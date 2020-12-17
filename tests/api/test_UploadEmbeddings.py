import unittest
import json
import os

import random
import responses
import numpy as np
import pytest

import torchvision
import tempfile
import lightly
from lightly.api import upload_embeddings_from_csv

@pytest.mark.slow
class TestUploadEmbeddings(unittest.TestCase):

    def setup(self, n_data=1000):
 
        self.dataset_id = 'XYZ'
        self.token = 'secret'

        # set up url
        self.dst_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                                  b'https://api.lightly.ai').decode()
        self.emb_url = f'{self.dst_url}/users/datasets/{self.dataset_id}/embeddings'
        self.tag_url = f'{self.dst_url}/users/datasets/{self.dataset_id}/tags/?token={self.token}'

        # create a dataset
        self.dataset = torchvision.datasets.FakeData(size=n_data,
                                                     image_size=(3, 32, 32))

        self.folder_path = tempfile.mkdtemp()
        self.path_to_embeddings = os.path.join(
            self.folder_path,
            'embeddings.csv'
        )

        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        labels = [0] * len(sample_names)

        lightly.utils.save_embeddings(
            self.path_to_embeddings,
            np.random.randn(n_data, 16),
            labels,
            sample_names
        )

    @responses.activate
    def test_upload_embeddings_no_tags(self):
        """Make sure upload of embeddings works.

        """

        self.setup(n_data=10000)

        def get_tags_callback(request):
            return (200, [], json.dumps([]))

        responses.add_callback(
            responses.GET, self.tag_url,
            callback=get_tags_callback,
            content_type='application/json'
        )

        with self.assertRaises(RuntimeError):
            success = upload_embeddings_from_csv(
                self.path_to_embeddings,
                dataset_id=self.dataset_id,
                token=self.token
            )
    
    @responses.activate
    def test_upload_embeddings_tag_exists(self):
        """

        """
        self.setup(n_data=10000)

        def get_tags_callback(request):
            return (200, [], json.dumps(['tag_1']))
        
        def get_embs_callback(request):
            return (200, [], json.dumps([{'name': 'default'}]))

        responses.add_callback(
            responses.GET, self.tag_url,
            callback=get_tags_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.GET, f'{self.emb_url}/?token={self.token}&mode=summaries',
            callback=get_embs_callback,
            content_type='application/json'
        )
    
        with self.assertRaises(RuntimeError):
            success = upload_embeddings_from_csv(
                self.path_to_embeddings,
                dataset_id=self.dataset_id,
                token=self.token
            )

    @responses.activate
    def test_upload_embeddings_with_tags(self):
        """Make sure upload of embeddings works.

        """

        self.setup(n_data=100)

        def get_tags_callback(request):
            return (200, [], json.dumps(['tag_1', 'tag_2']))

        def get_embs_callback(request):
            return (200, [], json.dumps([{'name': 'not_default'}]))
        
        def post_embedding_callback(request):
            return (200, [], json.dumps({}))

        responses.add_callback(
            responses.GET, self.tag_url,
            callback=get_tags_callback,
            content_type='application/json'
        )
    
        responses.add_callback(
            responses.GET, f'{self.emb_url}/?token={self.token}&mode=summaries',
            callback=get_embs_callback,
            content_type='application/json'
        )
    
        responses.add_callback(
            responses.POST, self.emb_url,
            callback=post_embedding_callback,
            content_type='application/json'
        )

        upload_embeddings_from_csv(
            self.path_to_embeddings,
            dataset_id=self.dataset_id,
            token=self.token
        )

