import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

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

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_all_video_frames(self):
        with tempfile.NamedTemporaryFile(suffix='.avi') as file:
            original = _generate_video(file.name)
            frames = list(download.download_all_video_frames(file.name))
            for frame, orig in zip(frames, original):
                assert _images_equal(frame, orig)

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_video_frame(self):
        with tempfile.NamedTemporaryFile(suffix='.avi') as file:
            original = _generate_video(file.name)
            for timestamp, orig in enumerate(original):
                frame = download.download_video_frame(file.name, timestamp)
                assert _images_equal(frame, orig)

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_video_frame_fps(self):
        for fps in range(1, 5):
            with self.subTest(msg=f"fps={fps}"), \
                tempfile.NamedTemporaryFile(suffix='.avi') as file:

                original = _generate_video(file.name, fps=fps)
                for timestamp, orig in enumerate(original):
                    frame = download.download_video_frame(file.name, timestamp / fps)
                    assert _images_equal(frame, orig)

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_video_frame_timestamp_exception(self):
        for fps in range(1, 5):
            with self.subTest(msg=f"fps={fps}"), \
                tempfile.NamedTemporaryFile(suffix='.avi') as file:

                original = _generate_video(file.name, fps=fps)

                # this should be the last frame and exist
                frame = download.download_video_frame(file.name, (len(original) - 1) / fps)
                assert _images_equal(frame, original[-1])

                # timestamp after last frame
                with self.assertRaises(ValueError):
                    download.download_video_frame(file.name, len(original) / fps)

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_video_frame_negative_timestamp_exception(self):
        for fps in range(1, 5):
            with self.subTest(msg=f"fps={fps}"), \
                tempfile.NamedTemporaryFile(suffix='.avi') as file:
                
                _generate_video(file.name, fps=fps)
                with self.assertRaises(ValueError):
                    download.download_video_frame(file.name, -1 / fps)

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

def _generate_video(out_file, n_frames=5, width=100, height=50, seed=0, fps=1):
    np.random.seed(seed)
    container = av.open(out_file, mode='w')
    stream = container.add_stream('libx264rgb', rate=fps)
    stream.width = width
    stream.height = height
    stream.options["crf"] = "0"
    stream.pix_fmt = "rgb24"
    images = (np.random.randn(n_frames, height, width, 3) * 255).astype(np.uint8)
    frames = [av.VideoFrame.from_ndarray(image, format='rgb24') for image in images]
    
    for frame in frames:
        for packet in stream.encode(frame):
            container.mux(packet)
        
    # flush and close
    packet = stream.encode(None)
    container.mux(packet)
    container.close()

    pil_images = [frame.to_image() for frame in frames]
    return pil_images
