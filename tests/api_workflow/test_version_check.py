import sys
import unittest
from time import sleep

from tqdm import tqdm


class TestVersionCheck(unittest.TestCase):


    def test_version_check_timeout(self):
        print("Starting test")

        def mocked_version_checking(self, current_version):
            sleep(10)
            print("Waiting endlessly.......")
            print("Because this code is protected by a try-catch, we need to call sys.exit()")
            sys.exit()

        import lightly.api.version_checking
        lightly.api.version_checking.VersioningApi.get_latest_pip_version = mocked_version_checking

        lightly.api.version_checking.get_latest_version("blub")

        print("Finished test")
