import unittest

import os
from PIL import Image

from PIL import Image

import lightly
from lightly.api.utils import retry
from lightly.api.utils import getenv
from lightly.api.utils import PIL_to_bytes

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

    def test_getenv(self):
        os.environ['TEST_ENV_VARIABLE'] = 'hello world'
        env = getenv('TEST_ENV_VARIABLE', 'default')
        self.assertEqual(env, 'hello world')

    def test_getenv_fail(self):
        env = getenv('TEST_ENV_VARIABLE_WHICH_DOES_NOT_EXIST', 'hello world')
        self.assertEqual(env, 'hello world')

    def test_PIL_to_bytes(self):
        image = Image.new('RGB', (128, 128))

        # test with quality=None
        PIL_to_bytes(image)

        # test with quality=90
        PIL_to_bytes(image, quality=90)

        # test with quality=90 and ext=jpg
        PIL_to_bytes(image, ext='JPEG', quality=90)
