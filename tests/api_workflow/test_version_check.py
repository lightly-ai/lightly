import os
import unittest
import time

import lightly
from lightly.api.version_checking import LightlyTimeoutException


class TestVersionCheck(unittest.TestCase):

    def assert_version_check_is_fast(self):
        start_time = time.time()

        with self.assertRaises(LightlyTimeoutException):
            lightly.do_version_check(lightly.__version__)

        duration = time.time() - start_time

        self.assertLess(duration, 1.5)

    def test_version_check_timeout_url(self):
        self.old_server_location = os.environ.get('LIGHTLY_SERVER_LOCATION', None)
        os.environ['LIGHTLY_SERVER_LOCATION'] = 'http://example.com:81'

        self.assert_version_check_is_fast()

        if self.old_server_location is not None:
            os.environ['LIGHTLY_SERVER_LOCATION'] = self.old_server_location

    def test_version_check_timout_mocked(self):

        def mocked_get_versioning_api_timeout():
            time.sleep(10)

        def mocked_get_versioning_api_error():
            raise ValueError

        mocked_get_versioning_api_functions = [
            mocked_get_versioning_api_error,
            mocked_get_versioning_api_timeout
        ]

        for i, mocked_get_versioning_api in enumerate(mocked_get_versioning_api_functions):
            with self.subTest(f"test_{i}"):

                self.assert_version_check_is_fast()

