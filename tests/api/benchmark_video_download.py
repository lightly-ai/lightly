import time
import unittest

import av
import numpy as np
from tqdm import tqdm

from lightly.api.download import download_video_frames_at_timestamps, \
    download_all_video_frames, download_video_frame

#@unittest.skip("Only used for benchmarks")
class BenchmarkDownloadVideoFrames(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.video_url_12min_100mb = "https://mediandr-a.akamaihd.net/progressive/2018/0912/TV-20180912-1628-0000.ln.mp4"
        with av.open(cls.video_url_12min_100mb ) as container:
            stream = container.streams.video[0]
            duration = stream.duration
            start_time = stream.start_time
            end_time = start_time + duration
        cls.timestamps = np.linspace(start_time, end_time, num=1000).astype(int).tolist()

    def setUp(self) -> None:
        self.start_time = time.time()

    def test_download_full(self):
        all_video_frames = download_all_video_frames(self.video_url_12min_100mb)
        for i, frame in enumerate(tqdm(all_video_frames)):
            pass

    # Takes very long for many frames, but is very quick for little frames
    # The reason is that
    # - every function call has quite some overhead
    # - as many frames are skipped by the seek, this only reads a little number of frames per function call.
    def test_download_at_timestamps_for_loop(self):
        for timestamp in tqdm(self.timestamps):
            frame = download_video_frame(self.video_url_12min_100mb, timestamp)

    def test_download_at_timestamps(self):
        frames = download_video_frames_at_timestamps(self.video_url_12min_100mb, self.timestamps)
        frames = list(tqdm(frames, total=len(self.timestamps)))

    # Takes long as it downloads the whole video first
    # Takes long, as it access the frames even at random locations, similar to
    # downloading specific frame in a for loop.
    def test_download_at_indices_decord(self):
        """
        See https://github.com/dmlc/decord/issues/199
        """
        import decord
        vr = decord.VideoReader(self.video_url_12min_100mb)
        decord.bridge.set_bridge('torch')
        print(f"Took {time.time() - self.start_time}s for creating the video reader.")
        frames = vr.get_batch(list(range(0, 18000, 18)))

    def tearDown(self) -> None:
        print(f"Took {time.time()-self.start_time}s")