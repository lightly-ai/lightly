import os
import sys
import json
import tempfile
import unittest
import warnings
from io import BytesIO
from unittest import mock

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

    @property
    def status_code(self):
        return 200

    def json(self):
        # instead of returning the byte stream from the url
        # we just load the json and return the dictionary
        with open(self._raw, 'r') as f:
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
        stream = open(self._raw, 'rb')
        if self.return_partial_stream:
            bytes = stream.read()
            stream_first_part = BytesIO(bytes[:1024])
            MockedResponsePartialStream.return_partial_stream = False
            return stream_first_part
        else:
            return stream


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

    @mock.patch("test_download.MockedResponse", MockedResponsePartialStream)
    def test_download_image_half_broken_retry_once(self):
        lightly.api.utils.RETRY_MAX_RETRIES = 1
        MockedResponse.return_partial_stream = True
        original = _pil_image()
        with tempfile.NamedTemporaryFile(suffix='.png') as file:
            original.save(file.name)
            # assert that the retry fails
            with self.assertRaises(RuntimeError):
                image = download.download_image(file.name)

    @mock.patch("test_download.MockedResponse", MockedResponsePartialStream)
    def test_download_image_half_broken_retry_twice(self):
        lightly.api.utils.RETRY_MAX_RETRIES = 2
        MockedResponse.return_partial_stream = True
        original = _pil_image()
        with tempfile.NamedTemporaryFile(suffix='.png') as file:
            original.save(file.name)
            image = download.download_image(file.name)
            assert _images_equal(image, original)


    def test_download_image(self):
        original = _pil_image()
        with tempfile.NamedTemporaryFile(suffix='.png') as file:
            original.save(file.name)
            image = download.download_image(file.name)
            assert _images_equal(image, original)

    def test_download_prediction(self):
        original = _json_prediction()
        with tempfile.NamedTemporaryFile(suffix='.json', mode="w+") as file:
            with open(file.name, 'w') as f:
                json.dump(original, f)
            response = download.download_prediction_file(file.name)
            self.assertDictEqual(response, original)

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
    def test_download_last_video_frame(self):
        with tempfile.NamedTemporaryFile(suffix='.avi') as file:
            n_frames = 5
            original = _generate_video(file.name, n_frames=n_frames)
            timestamps = list(range(1, n_frames+1))
            for timestamp in timestamps:
                with self.subTest(timestamp=timestamp):
                    if timestamp > n_frames:
                        with self.assertRaises(RuntimeError):
                            frame = download.download_video_frame(file.name, timestamp)
                    else:
                        frame = download.download_video_frame(file.name, timestamp)

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_video_frames_at_timestamps(self):
        with tempfile.NamedTemporaryFile(suffix='.avi') as file:
            n_frames = 5
            original = _generate_video(file.name, n_frames=n_frames)
            original_timestamps = list(range(1, n_frames+1))
            frame_indices = list(range(2, len(original) - 1, 2))
            timestamps = [original_timestamps[i] for i in frame_indices]
            frames = list(download.download_video_frames_at_timestamps(
                file.name, timestamps
            ))
            self.assertEqual(len(frames), len(timestamps))
            for frame, timestamp in zip(frames, frame_indices):
                orig = original[timestamp]
                assert _images_equal(frame, orig)

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_video_frames_at_timestamps_wrong_order(self):
        with tempfile.NamedTemporaryFile(suffix='.avi') as file:
            original = _generate_video(file.name)
            timestamps = [2, 1]
            with self.assertRaises(ValueError):
                frames = list(download.download_video_frames_at_timestamps(
                    file.name, timestamps
                ))

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_video_frames_at_timestamps_emtpy(self):
        with tempfile.NamedTemporaryFile(suffix='.avi') as file:
            frames = list(download.download_video_frames_at_timestamps(
                    file.name, timestamps=[]
                ))
            self.assertEqual(len(frames), 0)

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_all_video_frames_restart_throws(self):
        with tempfile.NamedTemporaryFile(suffix='.avi') as file:
            original = _generate_video(file.name)
            with self.assertRaises(ValueError):
                # timestamp too small
                frames = list(download.download_all_video_frames(file.name, timestamp=-1))
            with self.assertRaises(ValueError):
                # timestamp too large
                frames = list(download.download_all_video_frames(file.name, timestamp=6))

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_all_video_frames_restart_at_0(self):
        # relevant for restarting if the frame iterator is empty
        # although it shouldn't be
        with tempfile.NamedTemporaryFile(suffix='.avi') as file:
            original = _generate_video(file.name)
            frames = list(download.download_all_video_frames(file.name, timestamp=None))
            for frame, orig in zip(frames, original):
                assert _images_equal(frame, orig)

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_all_video_frames_restart(self):
        # relevant if decoding a frame goes wrong for some reason and we
        # want to try again
        restart_timestamp = 3
        with tempfile.NamedTemporaryFile(suffix='.avi') as file:
            original = _generate_video(file.name)
            frames = list(download.download_all_video_frames(file.name, restart_timestamp))
            for frame, orig in zip(frames, original[2:]):
                assert _images_equal(frame, orig)


    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_video_frame_fps(self):
        for fps in [24, 30, 60]:
            with self.subTest(msg=f"fps={fps}"), \
                tempfile.NamedTemporaryFile(suffix='.avi') as file:

                original = _generate_video(file.name, fps=fps)
                all_frames = download.download_all_video_frames(
                    file.name,
                    as_pil_image=False,
                )
                for true_frame in all_frames:
                    frame = download.download_video_frame(
                        file.name,
                        timestamp=true_frame.pts,
                        as_pil_image=False,
                    )
                    assert frame.pts == true_frame.pts

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_video_frame_timestamp_exception(self):
        for fps in [24, 30, 60]:
            with self.subTest(msg=f"fps={fps}"), \
                tempfile.NamedTemporaryFile(suffix='.avi') as file:

                original = _generate_video(file.name, fps=fps)

                # this should be the last frame and exist
                frame = download.download_video_frame(file.name, len(original))
                assert _images_equal(frame, original[-1])

                # timestamp after last frame
                with self.assertRaises(ValueError):
                    download.download_video_frame(file.name, len(original) + 1)

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_video_frame_negative_timestamp_exception(self):
        for fps in [24, 30, 60]:
            with self.subTest(msg=f"fps={fps}"), \
                tempfile.NamedTemporaryFile(suffix='.avi') as file:
                
                _generate_video(file.name, fps=fps)
                with self.assertRaises(ValueError):
                    download.download_video_frame(file.name, -1)

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
            download.download_and_write_all_files(
                file_infos, 
                output_dir=tempdir2, 
                max_workers=max_workers
            )

            for orig, filename in zip(originals, filenames):
                image = Image.open(os.path.join(tempdir2, filename))
                assert _images_equal(orig, image)

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_video_frame_count(self):
        fps = 24
        for true_n_frames in [24, 30, 60]:
            for suffix in ['.avi', '.mpeg']:
                with tempfile.NamedTemporaryFile(suffix=suffix) as file, \
                    self.subTest(msg=f'n_frames={true_n_frames}, extension={suffix}'):

                    _generate_video(file.name, n_frames=true_n_frames, fps=fps)
                    n_frames = download.video_frame_count(file.name)
                    assert n_frames == true_n_frames

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_video_frame_count_no_metadata(self):
        fps = 24
        for true_n_frames in [24, 30, 60]:
            for suffix in ['.avi', '.mpeg']:
                with tempfile.NamedTemporaryFile(suffix=suffix) as file, \
                    self.subTest(msg=f'n_frames={true_n_frames}, extension={suffix}'):

                    _generate_video(file.name, n_frames=true_n_frames, fps=fps)
                    n_frames = download.video_frame_count(
                        file.name,
                        ignore_metadata=True
                    )
                    assert n_frames == true_n_frames

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_all_video_frame_counts(self):
        true_n_frames = [3, 5]
        fps = 24
        for suffix in ['.avi', '.mpeg']:
            with tempfile.NamedTemporaryFile(suffix=suffix) as file1, \
                tempfile.NamedTemporaryFile(suffix=suffix) as file2, \
                self.subTest(msg=f'extension={suffix}'):

                _generate_video(file1.name, n_frames=true_n_frames[0], fps=fps)
                _generate_video(file2.name, n_frames=true_n_frames[1], fps=fps)
                frame_counts = download.all_video_frame_counts(
                    urls=[file1.name, file2.name],
                )
                assert sum(frame_counts) == sum(true_n_frames)
                assert frame_counts == true_n_frames

    @unittest.skipUnless(AV_AVAILABLE, "Pyav not installed")
    def test_download_all_video_frame_counts_broken(self):
        fps = 24
        n_frames = 5
        with tempfile.NamedTemporaryFile(suffix='.mpeg') as file1, \
            tempfile.NamedTemporaryFile(suffix='.mpeg') as file2:

            _generate_video(file1.name, fps=fps, n_frames=n_frames)
            _generate_video(file2.name, fps=fps, broken=True)
            
            urls = [file1.name, file2.name]
            result = download.all_video_frame_counts(urls)
            assert result == [n_frames, None]


def _images_equal(image1, image2):
    # note that images saved and loaded from disk must
    # use a lossless format, otherwise this equality will not hold
    return np.all(np.array(image1) == np.array(image2))

def _pil_image(width=100, height=50, seed=0):
    np.random.seed(seed)
    image = (np.random.randn(width, height, 3) * 255).astype(np.uint8)
    image = Image.fromarray(image, mode='RGB')
    return image

def _json_prediction():
    return {
        'string': 'Hello World',
        'int': 1,
        'float': 0.5,
    }

def _generate_video(
    out_file, 
    n_frames=5, 
    width=100, 
    height=50, 
    seed=0, 
    fps=24,
    broken=False,
):
    """Generate a video.

    Use .avi extension if you want to save a lossless video. Use '.mpeg' for
    videos which should have streams.frames = 0, so that the whole video must
    be loaded to find the total number of frames. Note that mpeg requires
    fps = 24.

    """
    is_mpeg = out_file.endswith('.mpeg')
    video_format = 'libx264rgb'
    pixel_format = 'rgb24'

    if is_mpeg:
        video_format = 'mpeg1video'
        pixel_format = 'yuv420p'

    if broken:
        n_frames = 0

    np.random.seed(seed)
    container = av.open(out_file, mode='w')
    stream = container.add_stream(video_format, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = pixel_format

    if is_mpeg:
        frames = [av.VideoFrame(width, height, pixel_format) for i in range(n_frames)]
    else:
        # save lossless video
        stream.options["crf"] = "0"
        images = (np.random.randn(n_frames, height, width, 3) * 255).astype(np.uint8)
        frames = [av.VideoFrame.from_ndarray(image, format=pixel_format) for image in images]
        
    for frame in frames:
        for packet in stream.encode(frame):
            container.mux(packet)

    if not broken:
        # flush the stream
        # video cannot be loaded if this is omitted
        packet = stream.encode(None)
        container.mux(packet)
        
    container.close()

    pil_images = [frame.to_image() for frame in frames]
    return pil_images
