import unittest

import os
from PIL import Image

from PIL import Image

import lightly
from lightly.api.utils import DatasourceType, get_signed_url_destination, retry
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

    def test_get_signed_url_destination(self):

        # S3
        self.assertEqual(
            get_signed_url_destination('https://lightly.s3.eu-central-1.amazonaws.com/lightly/somewhere/image.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=0123456789%2F20220811%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20220811T065010Z&X-Amz-Expires=601200&X-Amz-Signature=0123456789&X-Amz-SignedHeaders=host&x-id=GetObject'),
            DatasourceType.S3
        )
        self.assertNotEqual(
            get_signed_url_destination('http://someething.with.s3.in.it'),
            DatasourceType.S3
        )

        # GCS
        self.assertEqual(
            get_signed_url_destination('https://storage.googleapis.com/lightly/somewhere/image.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=lightly%40appspot.gserviceaccount.com%2F20220811%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220811T065325Z&X-Goog-Expires=601201&X-Goog-SignedHeaders=host&X-Goog-Signature=01234567890'),
            DatasourceType.GCS
        )
        self.assertNotEqual(
            get_signed_url_destination('http://someething.with.google.in.it'),
            DatasourceType.GCS
        )

        # AZURE
        self.assertEqual(
            get_signed_url_destination('https://lightly.blob.core.windows.net/lightly/somewhere/image.jpg?sv=2020-08-04&ss=bfqt&srt=sco&sp=0123456789&se=2022-04-13T20:20:02Z&st=2022-04-13T12:20:02Z&spr=https&sig=0123456789'),
            DatasourceType.AZURE
        )
        self.assertNotEqual(
            get_signed_url_destination('http://someething.with.windows.in.it'),
            DatasourceType.AZURE
        )
