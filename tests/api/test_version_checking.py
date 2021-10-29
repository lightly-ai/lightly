import sys
import time
import unittest

import lightly
from lightly.api.version_checking import get_latest_version, version_compare, \
    get_minimum_compatible_version, pretty_print_latest_version, \
    LightlyTimeoutException
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
        try:
            old_get_versioning_api = lightly.api.version_checking.get_versioning_api

            def mocked_get_versioning_api_timeout():
                time.sleep(10)
                print("This line should never be reached, calling sys.exti()")
                sys.exit()

            lightly.api.version_checking.get_versioning_api = mocked_get_versioning_api_timeout

            start_time = time.time()

            with self.assertRaises(LightlyTimeoutException):
                lightly.do_version_check(lightly.__version__)

            duration = time.time() - start_time

            self.assertLess(duration, 1.5)

        finally:
            lightly.api.version_checking.get_versioning_api = old_get_versioning_api
