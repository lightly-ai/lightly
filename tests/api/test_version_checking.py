import sys
import time
import unittest

import lightly
from lightly.api.version_checking import get_latest_version, version_compare, \
    get_minimum_compatible_version, pretty_print_latest_version, \
    LightlyTimeoutException, do_version_check
from tests.api_workflow.mocked_api_workflow_client import MockedVersioningApi


class TestVersionChecking(unittest.TestCase):

    def setUp(self) -> None:
        lightly.api.version_checking.VersioningApi = MockedVersioningApi

    def test_version_compare(self):
        assert version_compare("1.1.1", "2.2.2") < 0
        assert version_compare("1.1.1", "1.1.1") == 0
        assert version_compare("1.1.1", "0.0.0") > 0
        with self.assertRaises(AssertionError):
            version_compare("1.1", "1.1.1")
        with self.assertRaises(AssertionError):
            version_compare("1.1.1", "1.1")

    def test_get_latest_version(self):
        get_latest_version("1.2.3")

    def test_get_minimum_compatible_version(self):
        get_minimum_compatible_version()

    def test_pretty_print(self):
        pretty_print_latest_version(current_verion="curr", latest_version="1.1.1")

    def test_version_check_timout_mocked(self):
        """
            We cannot check for other errors as we don't know whether the
            current LIGHTLY_SERVER_URL is
            - unreachable (error in < 1 second)
            - causing a timeout and thus raising a LightlyTimeoutException
            - reachable (success in < 1 second

            Thus this only checks that the actual lightly.do_version_check()
            with needing >1s internally causes a LightlyTimeoutException
        """
        try:
            old_get_versioning_api = lightly.api.version_checking.get_versioning_api

            def mocked_get_versioning_api_timeout():
                time.sleep(10)
                print("This line should never be reached, calling sys.exti()")
                sys.exit()

            lightly.api.version_checking.get_versioning_api = mocked_get_versioning_api_timeout

            start_time = time.time()

            with self.assertRaises(LightlyTimeoutException):
                do_version_check(lightly.__version__)

            duration = time.time() - start_time

            self.assertLess(duration, 1.5)

        finally:
            lightly.api.version_checking.get_versioning_api = old_get_versioning_api
