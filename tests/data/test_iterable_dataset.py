import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import PIL
import torch

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


# Overwrite requests import in the light.data.iterable_dataset module.
import lightly

sys.modules["requests"] = MockedRequestsModule()
from lightly.data import iterable_dataset


class TestLightlyIterableDatasetHelperFunctions(unittest.TestCase):

    def test_top_directory(self):
        path_and_result = [
            ("", ""),
            ("/", ""),
            ("file.png", ""),
            ("/file.png", ""),
            ("top/file.png", "top"),
            ("/top/file.png", "top"),
            ("/top/dir/file.png", "top"),
        ]
        for path, result in path_and_result:
            top = iterable_dataset.top_directory(path)
            assert top == result
    
    def test_find_classes_from_filenames_flat(self):
        filename_and_class = [
            ('a.png', ""),
            ('b.png', "")
        ]
        filenames = [fname for fname, _ in filename_and_class]
        true_classes = [cls_ for _, cls_ in filename_and_class]
        classes, class_to_idx = iterable_dataset.find_classes_from_filenames(filenames)
        assert set(classes) == set(true_classes)

    def test_find_classes_from_filenames_nested_1(self):
        filename_and_class = [
            ('a/file.png', "a"),
            ('b/file.png', "b")
        ]
        filenames = [fname for fname, _ in filename_and_class]
        true_classes = [cls_ for _, cls_ in filename_and_class]
        classes, class_to_idx = iterable_dataset.find_classes_from_filenames(filenames)
        assert set(classes) == set(true_classes)

    def test_find_classes_from_filenames_nested_2(self):
        filename_and_class = [
            ('a/c/file.png', "a"),
            ('b/d/file.png', "b")
        ]
        filenames = [fname for fname, _ in filename_and_class]
        true_classes = [cls_ for _, cls_ in filename_and_class]
        classes, class_to_idx = iterable_dataset.find_classes_from_filenames(filenames)
        assert set(classes) == set(true_classes)

    def test_find_classes_from_filenames_absolute(self):
        filename_and_class = [
            ('/file.png', ""),
            ('/a/file.png', "a"),
            ('/b/c/file.png', "b"),
        ]
        filenames = [fname for fname, _ in filename_and_class]
        true_classes = [cls_ for _, cls_ in filename_and_class]
        classes, class_to_idx = iterable_dataset.find_classes_from_filenames(filenames)
        assert set(classes) == set(true_classes)
    
    def test_find_classes_from_filenames_special_cases(self):
        filename_and_class = [
            ('/', ""),
            ('', ""),
        ]
        filenames = [fname for fname, _ in filename_and_class]
        true_classes = [cls_ for _, cls_ in filename_and_class]
        classes, class_to_idx = iterable_dataset.find_classes_from_filenames(filenames)
        assert set(classes) == set(true_classes)

    def test_find_classes_from_filenames_index(self):
        filename_and_class = [
            ('a/file.png', "a"),
            ('b/file.png', "b"),
            ('c/d/file.png', "c"),
            ('/d/file.png', "d"),
            ('/e/f/file.png', "e"),
        ]
        filenames = [fname for fname, _ in filename_and_class]
        true_class_to_idx = {cls_: idx for idx, (_, cls_) in enumerate(filename_and_class)}
        # reverse to avoid passing already sorted array
        filename_and_class = list(reversed(filename_and_class))
        classes, class_to_idx = iterable_dataset.find_classes_from_filenames(filenames)
        assert set(classes) == set(class_to_idx.keys())
        assert set(class_to_idx.values()) == set(true_class_to_idx.values())
        for _, true_cls in filename_and_class:
            assert class_to_idx[true_cls] == true_class_to_idx[true_cls]


class TestLightlyImageIterableDataset(unittest.TestCase):

    def create_dataset(self, tempdir, n_images=5, n_dirs=0):
        file_idx = 0
        samples = []
        images = []

        def create_images(dirname, file_idx):
            for _ in range(n_images):
                filename = f"file_{file_idx}.png"
                filename = os.path.join(dirname, filename)
                filepath = os.path.join(tempdir, filename)
                image = _pil_image(seed=file_idx)
                image.save(filepath)
                samples.append((filename, filepath))
                images.append(image)
                file_idx += 1
            return file_idx
        
        for dir_idx in range(n_dirs):
            dirname = f"dir_{dir_idx}"
            dirpath = pathlib.Path(os.path.join(tempdir, dirname))
            dirpath.mkdir()
            file_idx = create_images(dirname, file_idx)
                
        if n_dirs == 0:
            create_images("", file_idx)

        return samples, images

    def test_lightly_image_iterable_dataset_create_flat(self):
        n_images = 5
        with tempfile.TemporaryDirectory() as tempdir:
            samples, images = self.create_dataset(tempdir, n_images)    
            dataset = iterable_dataset.LightlyImageIterableDataset(samples)
            assert len(dataset.samples) == len(samples)
            assert len(dataset.classes) == 1
            assert len(dataset.class_to_idx) == 1

    def test_lightly_image_iterable_dataset_create_deep(self):
        n_images = 5
        n_dirs = 4
        with tempfile.TemporaryDirectory() as tempdir:
            samples, images = self.create_dataset(tempdir, n_images, n_dirs)
            dataset = iterable_dataset.LightlyImageIterableDataset(samples)
            print(samples)
            assert len(dataset.samples) == len(samples)
            assert len(dataset.classes) == n_dirs
            assert len(dataset.class_to_idx) == n_dirs
            assert min(dataset.class_to_idx.values()) == 0
            assert max(dataset.class_to_idx.values()) == n_dirs - 1
            
    def test_lightly_image_iterable_dataset_iter_flat(self):
        n_images = 5
        with tempfile.TemporaryDirectory() as tempdir:
            samples, images = self.create_dataset(tempdir, n_images)
            dataset = iterable_dataset.LightlyImageIterableDataset(samples)
            for i, (img, fname, cls_) in enumerate(dataset):
                with self.subTest(msg=f"i={i}"):
                    true_fname = samples[i][0]
                    true_image = images[i]
                    assert fname == true_fname
                    assert _images_equal(img, true_image)
                    assert cls_ == i // n_images

    def test_lightly_image_iterable_dataset_iter_nested(self):
        n_images = 5
        n_dirs = 4
        with tempfile.TemporaryDirectory() as tempdir:
            samples, images = self.create_dataset(tempdir, n_images, n_dirs)
            dataset = iterable_dataset.LightlyImageIterableDataset(samples)
            for i, (img, fname, cls_) in enumerate(dataset):
                with self.subTest(msg=f"i={i}"):
                    true_fname = samples[i][0]
                    true_image = images[i]
                    assert fname == true_fname
                    assert _images_equal(img, true_image)
                    assert cls_ == i // n_images

    def test_lightly_image_iterable_dataset_dataloader(self):
        n_images = 15
        batch_size = 3
        with tempfile.TemporaryDirectory() as tempdir:
            samples, images = self.create_dataset(tempdir, n_images)
            dataset = iterable_dataset.LightlyImageIterableDataset(samples)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                num_workers=2,
                batch_size=batch_size,
                collate_fn=lambda x: x,
            )
            for _ in enumerate(dataloader):
                pass


class TestLightlyVideoIterableDataset(unittest.TestCase):

    def create_dataset(self, tempdir, n_videos=3, n_frames=5, nested=False):
        frames = []
        samples = []
        for video_idx in range(n_videos):
            video_name = f"video_{video_idx}.avi"
            if nested:
                # lets put every video into a different directory
                dir_name = f"dir_{video_idx}"
                os.mkdir(os.path.join(tempdir, dir_name))
                video_name = os.path.join(dir_name, video_name)
            video_path = os.path.join(tempdir, video_name)
            video_frames = _generate_video(video_path, n_frames, seed=video_idx)
            frames.extend(video_frames)
            samples.append((video_name, video_path))
        return samples, frames

    def test_lightly_video_iterable_dataset_create_flat(self):
        n_videos = 3
        with tempfile.TemporaryDirectory() as tempdir:
            samples, _ = self.create_dataset(tempdir, n_videos)
            video_names = [vname for vname, _ in samples]
            dataset = iterable_dataset.LightlyVideoIterableDataset(samples)
            assert len(dataset.samples) == n_videos
            assert len(dataset.classes) == n_videos
            assert len(dataset.class_to_idx) == n_videos
            assert min(dataset.class_to_idx.values()) == 0
            assert max(dataset.class_to_idx.values()) == n_videos - 1
            assert set(dataset.classes) == set(video_names)

    def test_lightly_video_iterable_dataset_create_nested(self):
        n_videos = 3
        with tempfile.TemporaryDirectory() as tempdir:
            samples, _ = self.create_dataset(tempdir, n_videos, nested=True)
            video_names = [vname for vname, _ in samples]
            dataset = iterable_dataset.LightlyVideoIterableDataset(samples)
            assert len(dataset.samples) == n_videos
            assert len(dataset.classes) == n_videos
            assert len(dataset.class_to_idx) == n_videos
            assert min(dataset.class_to_idx.values()) == 0
            assert max(dataset.class_to_idx.values()) == n_videos - 1
            assert set(dataset.classes) == set(video_names)

    def test_lightly_video_iterable_dataset_iter_flat(self):
        n_videos = 3
        n_frames = 5
        with tempfile.TemporaryDirectory() as tempdir:
            samples, frames = self.create_dataset(tempdir, n_videos, n_frames)
            dataset = iterable_dataset.LightlyVideoIterableDataset(samples)
            for i, (frame, frame_name, target) in enumerate(dataset):
                with self.subTest(msg=f"i={i}"):
                    true_frame = frames[i]
                    true_target = i // n_frames
                    true_video_name = samples[i // n_frames][0]
                    # remove file extension
                    true_video_name, _ = os.path.splitext(true_video_name)
                    assert _images_equal(frame, true_frame)
                    assert target == true_target
                    print(frame_name)
                    print(true_video_name)
                    assert frame_name.startswith(true_video_name)

    


def _images_equal(image1, image2):
    # note that images saved and loaded from disk must
    # use a lossless format, otherwise this equality will not hold
    return np.all(np.array(image1) == np.array(image2))

def _pil_image(width=100, height=50, seed=0):
    np.random.seed(seed)
    image = (np.random.randn(width, height, 3) * 255).astype(np.uint8)
    image = PIL.Image.fromarray(image, mode='RGB')
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
