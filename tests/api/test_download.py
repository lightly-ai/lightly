import json
import os
import sys
import tempfile
import unittest
import warnings
from io import BytesIO
from unittest import mock

import numpy as np
import tqdm
from PIL import Image


# mock requests module so that files are read from
# disk instead of loading them from a remote url


class MockedRequestsModule:
    def get(self, url, stream=None, *args, **kwargs):
        return MockedResponse(url)

    class Session:
        def get(self, url, stream=None, *args, **kwargs):
            return MockedResponse(url)


class MockedRequestsModulePartialResponse:
    def get(self, url, stream=None, *args, **kwargs):
        return MockedResponsePartialStream(url)

    def raise_for_status(self):
        return

    class Session:
        def get(self, url, stream=None, *args, **kwargs):
            return MockedResponsePartialStream(url)


class MockedResponse:
    def __init__(self, raw):
        self._raw = raw

    @property
    def raw(self):
        # instead of returning the byte stream from the url
        # we just give back an openend filehandle
        return open(self._raw, "rb")

    @property
    def status_code(self):
        return 200

    def raise_for_status(self):
        return

    def json(self):
        # instead of returning the byte stream from the url
        # we just load the json and return the dictionary
        with open(self._raw, "r") as f:
            return json.load(f)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockedResponsePartialStream(MockedResponse):
    return_partial_stream = True

    @property
    def raw(self):
        # instead of returning the byte stream from the url
        # we just give back an openend filehandle
        stream = open(self._raw, "rb")
        if self.return_partial_stream:
            bytes = stream.read()
            stream_first_part = BytesIO(bytes[:1024])
            MockedResponsePartialStream.return_partial_stream = False
            return stream_first_part
        else:
            return stream


import lightly


@mock.patch("lightly.api.download.requests", MockedRequestsModulePartialResponse())
class TestDownloadPartialRespons(unittest.TestCase):
    def setUp(self):
        self._max_retries = lightly.api.utils.RETRY_MAX_RETRIES
        self._max_backoff = lightly.api.utils.RETRY_MAX_BACKOFF
        lightly.api.utils.RETRY_MAX_RETRIES = 1
        lightly.api.utils.RETRY_MAX_BACKOFF = 0
        warnings.filterwarnings("ignore")

    def tearDown(self):
        lightly.api.utils.RETRY_MAX_RETRIES = self._max_retries
        lightly.api.utils.RETRY_MAX_BACKOFF = self._max_backoff
        warnings.filterwarnings("default")

    def test_download_image_half_broken_retry_once(self):
        lightly.api.utils.RETRY_MAX_RETRIES = 1

        original = _pil_image()
        with tempfile.NamedTemporaryFile(suffix=".png") as file:
            original.save(file.name)
            # assert that the retry fails
            with self.assertRaises(RuntimeError) as error:
                image = lightly.api.download.download_image(file.name)
            self.assertTrue("Maximum retries exceeded" in str(error.exception))
            self.assertTrue("<class 'OSError'>" in str(error.exception))
            self.assertTrue("image file is truncated" in str(error.exception))

    def test_download_image_half_broken_retry_twice(self):
        lightly.api.utils.RETRY_MAX_RETRIES = 2
        MockedResponse.return_partial_stream = True
        original = _pil_image()
        with tempfile.NamedTemporaryFile(suffix=".png") as file:
            original.save(file.name)
            image = lightly.api.download.download_image(file.name)
            assert _images_equal(image, original)


@mock.patch("lightly.api.download.requests", MockedRequestsModule())
class TestDownload(unittest.TestCase):
    def setUp(self):
        self._max_retries = lightly.api.utils.RETRY_MAX_RETRIES
        self._max_backoff = lightly.api.utils.RETRY_MAX_BACKOFF
        lightly.api.utils.RETRY_MAX_RETRIES = 1
        lightly.api.utils.RETRY_MAX_BACKOFF = 0
        warnings.filterwarnings("ignore")

    def tearDown(self):
        lightly.api.utils.RETRY_MAX_RETRIES = self._max_retries
        lightly.api.utils.RETRY_MAX_BACKOFF = self._max_backoff
        warnings.filterwarnings("default")

    def test_download_image(self):
        original = _pil_image()
        with tempfile.NamedTemporaryFile(suffix=".png") as file:
            original.save(file.name)
            for request_kwargs in [None, {"stream": False}]:
                with self.subTest(request_kwargs=request_kwargs):
                    image = lightly.api.download.download_image(
                        file.name, request_kwargs=request_kwargs
                    )
                    assert _images_equal(image, original)

    def test_download_prediction(self):
        original = _json_prediction()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as file:
            with open(file.name, "w") as f:
                json.dump(original, f)
            for request_kwargs in [None, {"stream": False}]:
                with self.subTest(request_kwargs=request_kwargs):
                    response = lightly.api.download.download_prediction_file(
                        file.name,
                        request_kwargs=request_kwargs,
                    )
                    self.assertDictEqual(response, original)

    def test_download_image_with_session(self):
        session = MockedRequestsModule.Session()
        original = _pil_image()
        with tempfile.NamedTemporaryFile(suffix=".png") as file:
            original.save(file.name)
            image = lightly.api.download.download_image(file.name, session=session)
            assert _images_equal(image, original)

    def test_download_and_write_file(self):
        original = _pil_image()
        with tempfile.NamedTemporaryFile(
            suffix=".png"
        ) as file1, tempfile.NamedTemporaryFile(suffix=".png") as file2:
            original.save(file1.name)
            lightly.api.download.download_and_write_file(file1.name, file2.name)
            image = Image.open(file2.name)
            assert _images_equal(original, image)

    def test_download_and_write_file_with_session(self):
        session = MockedRequestsModule.Session()
        original = _pil_image()
        with tempfile.NamedTemporaryFile(
            suffix=".png"
        ) as file1, tempfile.NamedTemporaryFile(suffix=".png") as file2:
            original.save(file1.name)
            lightly.api.download.download_and_write_file(
                file1.name, file2.name, session=session
            )
            image = Image.open(file2.name)
            assert _images_equal(original, image)

    def test_download_and_write_all_files(self):
        n_files = 3
        max_workers = 2
        originals = [_pil_image(seed=i) for i in range(n_files)]
        filenames = [f"filename_{i}.png" for i in range(n_files)]
        with tempfile.TemporaryDirectory() as tempdir1, tempfile.TemporaryDirectory() as tempdir2:
            for request_kwargs in [None, {"stream": False}]:
                with self.subTest(request_kwargs=request_kwargs):
                    # save images at "remote" location
                    urls = [
                        os.path.join(tempdir1, f"url_{i}.png") for i in range(n_files)
                    ]
                    for image, url in zip(originals, urls):
                        image.save(url)

                    # download images from remote to local
                    file_infos = list(zip(filenames, urls))
                    lightly.api.download.download_and_write_all_files(
                        file_infos,
                        output_dir=tempdir2,
                        max_workers=max_workers,
                        request_kwargs=request_kwargs,
                    )

                    for orig, filename in zip(originals, filenames):
                        image = Image.open(os.path.join(tempdir2, filename))
                        assert _images_equal(orig, image)


def _images_equal(image1, image2):
    # note that images saved and loaded from disk must
    # use a lossless format, otherwise this equality will not hold
    return np.all(np.array(image1) == np.array(image2))


def _pil_image(width=100, height=50, seed=0):
    np.random.seed(seed)
    image = (np.random.randn(width, height, 3) * 255).astype(np.uint8)
    image = Image.fromarray(image, mode="RGB")
    return image


def _json_prediction():
    return {
        "string": "Hello World",
        "int": 1,
        "float": 0.5,
    }
