import unittest

import lightly
from lightly.api.version_checking import get_latest_version, version_compare, \
    get_minimum_compatible_version, pretty_print_latest_version
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
        pretty_print_latest_version("1.1.1")
