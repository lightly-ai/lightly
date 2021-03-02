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
