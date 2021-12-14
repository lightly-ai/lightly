import os
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

# mock requests module so that files are read from 
# disk instead of loading them from a remote url

class MockedRequestsModule:

    def get(self, url, stream=None):
        return MockedResponse(url)

    class Session:
        def get(self, url, stream=None):
            return MockedResponse(url)
    
class MockedResponse:

    def __init__(self, raw):
        self._raw = raw

    @property
    def raw(self):
        # instead of returning the byte stream from the url
        # we just give back an openend filehandle
        return open(self._raw, 'rb')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass 

# Overwrite requests import in the light.api.download module.
# lightly.api must be imported before because otherwise it
# will be loaded by lightly.api.download and use the mocked
# requests module instead of the real one.
import lightly.api

requests = sys.modules["requests"]
sys.modules["requests"] = MockedRequestsModule()
from lightly.api import download

sys.modules["requests"] = requests


class TestDownload(unittest.TestCase):

    def test_download_image(self):
        original = _pil_image()
        with tempfile.NamedTemporaryFile(suffix='.png') as file:
            original.save(file.name)
            image = download.download_image(file.name)
            assert _images_equal(image, original)

    def test_download_image_with_session(self):
        session = MockedRequestsModule.Session()
        original = _pil_image()
        with tempfile.NamedTemporaryFile(suffix='.png') as file:
            original.save(file.name)
            image = download.download_image(file.name, session=session)
            assert _images_equal(image, original)

    def test_download_all_video_frames(self):
        with tempfile.NamedTemporaryFile(suffix='.avi') as file:
            original = _generate_video(file.name)
            frames = list(download.download_all_video_frames(file.name, as_pil_image=False))
            for frame, orig in zip(frames, original):
                assert _images_equal(frame, orig)

    def test_download_and_write_file(self):
        original = _pil_image()
        with tempfile.NamedTemporaryFile(suffix='.png') as file1, \
            tempfile.NamedTemporaryFile(suffix='.png') as file2:
            
            original.save(file1.name)
            download.download_and_write_file(file1.name, file2.name)
            image = Image.open(file2.name)
            assert _images_equal(original, image)
    
    def test_download_and_write_file_with_session(self):
        session = MockedRequestsModule.Session()
        original = _pil_image()
        with tempfile.NamedTemporaryFile(suffix='.png') as file1, \
            tempfile.NamedTemporaryFile(suffix='.png') as file2:
            
            original.save(file1.name)
            download.download_and_write_file(file1.name, file2.name, session=session)
            image = Image.open(file2.name)
            assert _images_equal(original, image)

    def test_download_and_write_all_files(self):
        n_files = 3
        max_workers = 2
        originals = [_pil_image(seed=i) for i in range(n_files)]
        filenames = [f'filename_{i}.png' for i in range(n_files)]
        with tempfile.TemporaryDirectory() as tempdir1, \
            tempfile.TemporaryDirectory() as tempdir2:

            # save images at "remote" location
            urls = [os.path.join(tempdir1, f'url_{i}.png') for i in range(n_files)]
            for image, url in zip(originals, urls):
                image.save(url)

            # download images from remote to local
            file_infos = list(zip(filenames, urls))
            download.download_and_write_all_files(file_infos, output_dir=tempdir2, max_workers=max_workers)

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
    image = Image.fromarray(image, mode='RGB')
    return image

def _generate_video(out_file, n_frames=5, width=100, height=50, codec=cv2.VideoWriter_fourcc(*'LAGS'), seed=0):
    np.random.seed(seed)
    frames = (np.random.randn(n_frames, width, height, 3) * 255).astype(np.uint8)
    out = cv2.VideoWriter(out_file, codec, 0, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return frames


