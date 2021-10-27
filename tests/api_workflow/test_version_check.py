import os
import sys
import unittest
import time

import lightly
from lightly.api.version_checking import LightlyTimeoutException


class TestVersionCheck(unittest.TestCase):
    """
    We cannot check for other errors as we don't know whether the
    current LIGHTLY_SERVER_URL is
    - unreachable (error in < 1 second)
    - causing a timeout and thus raising a LightlyTimeoutException
    - reachable (success in < 1 second
    """

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




