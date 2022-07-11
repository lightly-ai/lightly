import tempfile
import random
from typing import Tuple, List
import unittest

from lightly.data.dataset import LightlyDataset
from lightly.data.lightly_subset import LightlySubset

from tests.data.test_LightlyDataset import TestLightlyDataset

try:
    from lightly.data._video import VideoDataset    
    import av
    import cv2
    VIDEO_DATASET_AVAILABLE = True
except ModuleNotFoundError:
    VIDEO_DATASET_AVAILABLE = False

class TestLightlySubset(TestLightlyDataset):
    def setUp(self) -> None:
        tmp_dir, folder_names, sample_names = self.create_dataset(n_subfolders=5, n_samples_per_subfolder=5)
        self.input_dir = tmp_dir
        
    def create_subset(self, seed=0) -> Tuple[LightlySubset, List[str]]:
        random.seed(seed)
        base_dataset = LightlyDataset(input_dir=self.input_dir)
        filenames_base_dataset = base_dataset.get_filenames()

        no_samples_subset = int(len(filenames_base_dataset) * 0.5)
        filenames_subset = random.sample(filenames_base_dataset, no_samples_subset)

        subset = LightlySubset(base_dataset=base_dataset, filenames_subset=filenames_subset)
        return subset, filenames_subset

    @unittest.skipUnless(VIDEO_DATASET_AVAILABLE, "PyAV and CV2 are both installed")
    def create_video_subset(self, seed=0) -> Tuple[LightlySubset, List[str]]:
        random.seed(seed)
        self.create_video_dataset(n_videos=5, n_frames_per_video=10)
        base_dataset = LightlyDataset(self.input_dir)
        filenames_base_dataset = base_dataset.get_filenames()

        no_samples_subset = int(len(filenames_base_dataset) * 0.5)
        filenames_subset = random.sample(filenames_base_dataset, no_samples_subset)

        subset = LightlySubset(base_dataset=base_dataset, filenames_subset=filenames_subset)
        return subset, filenames_subset

    def test_create_lightly_subset(self):
        subset, filenames_subset = self.create_subset()
        
        assert subset.get_filenames() == filenames_subset
        for index_subset, filename_subset in enumerate(filenames_subset):
            sample, target, fname = subset.__getitem__(index_subset)
            assert filename_subset == fname

    @unittest.skipUnless(VIDEO_DATASET_AVAILABLE, "PyAV and CV2 are both installed")
    def test_create_lightly_video_subset(self):
        subset, filenames_subset = self.create_video_subset()
        
        assert subset.get_filenames() == filenames_subset
        for index_subset, filename_subset in enumerate(filenames_subset):
            sample, target, fname = subset.__getitem__(index_subset)
            assert filename_subset == fname
            
    def test_lightly_subset_transform(self):
        subset, filenames_subset = self.create_subset()
        self.test_transform_setter(dataset=subset)

    def test_lightly_subset_dump(self):
        subset, filenames_subset = self.create_subset()
        dataset = subset

        out_dir = tempfile.mkdtemp()
        dataset.dump(out_dir)

        files_output_dir = LightlyDataset(input_dir=out_dir).get_filenames()
        assert set(files_output_dir) == set(dataset.get_filenames())
