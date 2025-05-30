import os
import unittest
from unittest import mock

import pytest
from PIL import Image

from lightly.api.utils import (
    DatasourceType,
    PIL_to_bytes,
    get_lightly_server_location_from_env,
    get_signed_url_destination,
    getenv,
    paginate_endpoint,
)


class TestUtils(unittest.TestCase):
    def test_getenv(self):
        os.environ["TEST_ENV_VARIABLE"] = "hello world"
        env = getenv("TEST_ENV_VARIABLE", "default")
        self.assertEqual(env, "hello world")

    def test_getenv_fail(self):
        env = getenv("TEST_ENV_VARIABLE_WHICH_DOES_NOT_EXIST", "hello world")
        self.assertEqual(env, "hello world")

    def test_PIL_to_bytes(self):
        image = Image.new("RGB", (128, 128))

        # test with quality=None
        PIL_to_bytes(image)

        # test with quality=90
        PIL_to_bytes(image, quality=90)

        # test with quality=90 and ext=jpg
        PIL_to_bytes(image, ext="JPEG", quality=90)

    def test_get_signed_url_destination(self):
        # S3
        self.assertEqual(
            get_signed_url_destination(
                "https://lightly.s3.eu-central-1.amazonaws.com/lightly/somewhere/image.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=0123456789%2F20220811%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20220811T065010Z&X-Amz-Expires=601200&X-Amz-Signature=0123456789&X-Amz-SignedHeaders=host&x-id=GetObject"
            ),
            DatasourceType.S3,
        )
        self.assertNotEqual(
            get_signed_url_destination("http://someething.with.s3.in.it"),
            DatasourceType.S3,
        )

        # GCS
        self.assertEqual(
            get_signed_url_destination(
                "https://storage.googleapis.com/lightly/somewhere/image.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=lightly%40appspot.gserviceaccount.com%2F20220811%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220811T065325Z&X-Goog-Expires=601201&X-Goog-SignedHeaders=host&X-Goog-Signature=01234567890"
            ),
            DatasourceType.GCS,
        )
        self.assertNotEqual(
            get_signed_url_destination("http://someething.with.google.in.it"),
            DatasourceType.GCS,
        )

        # AZURE
        self.assertEqual(
            get_signed_url_destination(
                "https://lightly.blob.core.windows.net/lightly/somewhere/image.jpg?sv=2020-08-04&ss=bfqt&srt=sco&sp=0123456789&se=2022-04-13T20:20:02Z&st=2022-04-13T12:20:02Z&spr=https&sig=0123456789"
            ),
            DatasourceType.AZURE,
        )
        self.assertNotEqual(
            get_signed_url_destination("http://someething.with.windows.in.it"),
            DatasourceType.AZURE,
        )

    def test_get_lightly_server_location_from_env(self):
        os.environ["LIGHTLY_SERVER_LOCATION"] = "https://api.dev.lightly.ai/ "
        host = get_lightly_server_location_from_env()
        self.assertEqual(host, "https://api.dev.lightly.ai")

    def test_paginate_endpoint(self):
        def some_function(page_size=8, page_offset=0):
            if page_offset > 3 * page_size:
                assert False  # should not happen
            elif page_offset > 2 * page_size:
                return (page_size - 1) * ["a"]
            else:
                return page_size * ["a"]

        page_size = 8
        some_iterator = paginate_endpoint(some_function, page_size=page_size)
        some_list = list(some_iterator)
        self.assertEqual((4 * page_size - 1) * ["a"], some_list)
        self.assertEqual(len(some_list), (4 * page_size - 1))

    def test_paginate_endpoint__string(self):
        def paginated_function(page_size=8, page_offset=0):
            """Returns one page of size page_size, then one page of size page_size - 1."""
            if page_offset > 3 * page_size:
                assert False  # This should not happen.
            elif page_offset > 2 * page_size:
                return (page_size - 1) * "a"
            else:
                return page_size * "a"

        page_size = 8
        some_iterator = paginate_endpoint(paginated_function, page_size=page_size)
        some_list = list(some_iterator)
        self.assertEqual((4 * page_size - 1) * "a", "".join(some_list))
        self.assertEqual(len(some_list), 4)  # Expect four pages of strings.

    def test_paginate_endpoint__multiple_of_page_size(self):
        def paginated_function(page_size=8, page_offset=0):
            """Returns two pages of size page_size, then an empty page."""
            if page_offset > 3 * page_size:
                return []
            elif page_offset > 2 * page_size:
                return page_size * ["a"]
            else:
                return page_size * ["a"]

        page_size = 8
        some_iterator = paginate_endpoint(paginated_function, page_size=page_size)
        some_list = list(some_iterator)
        self.assertEqual((4 * page_size) * ["a"], some_list)
        self.assertEqual(len(some_list), (4 * page_size))

    def test_paginate_endpoint_empty(self):
        def some_function(page_size=8, page_offset=0):
            return []

        some_iterator = paginate_endpoint(some_function, page_size=8)
        some_list = list(some_iterator)
        self.assertEqual(some_list, [])
