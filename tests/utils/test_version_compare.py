import unittest

from lightly.utils import version_compare


class TestVersionCompare(unittest.TestCase):

    def test_valid_versions(self):

        # general test of smaller than version numbers
        self.assertEqual(version_compare.version_compare('0.1.4', '1.2.0'), -1)
        self.assertEqual(version_compare.version_compare('1.1.0', '1.2.0'), -1)

        # test bigger than
        self.assertEqual(version_compare.version_compare('1.2.0', '1.1.0'), 1)
        self.assertEqual(version_compare.version_compare('1.2.0', '0.1.4'), 1)

        # test equal
        self.assertEqual(version_compare.version_compare('1.2.0', '1.2.0'), 0)


    def test_invalid_versions(self):
        with self.assertRaises(ValueError):
            version_compare.version_compare('1.2', '1.1.0')

        with self.assertRaises(ValueError):
            version_compare.version_compare('1.2.0.1', '1.1.0')

        # test within same minor version and with special cases
        with self.assertRaises(ValueError):
            self.assertEqual(version_compare.version_compare('1.0.7', '1.1.0.dev1'), -1)

        with self.assertRaises(ValueError):
            self.assertEqual(version_compare.version_compare('1.1.0.dev1', '1.1.0rc1'), -1)