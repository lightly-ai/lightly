import unittest
import json
import os
import pytest

from lightly.api import routes

@pytest.mark.slow
class TestRoutes(unittest.TestCase):

    def test_routes(self):

        dataset_id = 'XYZ'
        sample_id = 'xyz'
        tag_id = 'abc'

        dst_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                             b'https://api.lightly.ai').decode()

        # pip
        self.assertEqual(
            routes.pip.service._prefix(),
            f'{dst_url}/pip'
        )
        # users
        self.assertEqual(
            routes.users.service._prefix(),
            f'{dst_url}/users'
        )
        # datasets
        self.assertEqual(
            routes.users.datasets.service._prefix(),
            f'{dst_url}/users/datasets'
        )
        self.assertEqual(
            routes.users.datasets.service._prefix(dataset_id=dataset_id),
            f'{dst_url}/users/datasets/{dataset_id}'
        )
        # embeddings
        self.assertEqual(
            routes.users.datasets.embeddings.service._prefix(),
            f'{dst_url}/users/datasets/embeddings'
        )
        self.assertEqual(
            routes.users.datasets.embeddings.service._prefix(dataset_id=dataset_id),
            f'{dst_url}/users/datasets/{dataset_id}/embeddings'
        )
        # samples
        self.assertEqual(
            routes.users.datasets.samples.service._prefix(),
            f'{dst_url}/users/datasets/samples'
        )
        self.assertEqual(
            routes.users.datasets.samples.service._prefix(dataset_id=dataset_id),
            f'{dst_url}/users/datasets/{dataset_id}/samples'
        )
        self.assertEqual(
            routes.users.datasets.samples.service._prefix(dataset_id=dataset_id, sample_id=sample_id),
            f'{dst_url}/users/datasets/{dataset_id}/samples/{sample_id}'
        )
        # tags
        self.assertEqual(
            routes.users.datasets.tags.service._prefix(),
            f'{dst_url}/users/datasets/tags'
        )
        self.assertEqual(
            routes.users.datasets.tags.service._prefix(dataset_id=dataset_id),
            f'{dst_url}/users/datasets/{dataset_id}/tags'
        )
        self.assertEqual(
            routes.users.datasets.tags.service._prefix(dataset_id=dataset_id, tag_id=tag_id),
            f'{dst_url}/users/datasets/{dataset_id}/tags/{tag_id}'
        )
