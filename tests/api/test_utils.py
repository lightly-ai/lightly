import unittest

import lightly
from lightly.api.utils import retry


class TestUtils(unittest.TestCase):

    def test_retry_success(self):

        def my_func(arg, kwarg=5):
            return arg + kwarg
        
        self.assertEqual(retry(my_func, 5, kwarg=5), 10)


    def test_retry_fail(self):

        def my_func():
            raise RuntimeError()
        
        with self.assertRaises(RuntimeError):
            retry(my_func)
