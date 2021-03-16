from typing import List, Dict, Tuple

from lightly.data.dataset import LightlyDataset


class LightlySubset(LightlyDataset):
    def __init__(self, base_dataset: LightlyDataset, filenames_subset: List[str]):
        self.base_dataset = base_dataset
        self.filenames_subset = filenames_subset

        dict_base_dataset_filename_index: Dict[str, int] = dict()
        for index in range(len(base_dataset)):
            fname = base_dataset.index_to_filename(self.dataset, index)
            dict_base_dataset_filename_index[fname] = index

        self.mapping_subset_index_to_baseset_index = \
            [dict_base_dataset_filename_index[filename] for filename in filenames_subset]

    def __getitem__(self, index_subset: int) -> Tuple[object, object, str]:
        index_baseset = self.mapping_subset_index_to_baseset_index[index_subset]
        sample, target, fname = self.base_dataset.__getitem__(index_baseset)
        return sample, target, fname

    def __len__(self) -> int:
        return len(self.mapping_subset_index_to_baseset_index)

    def index_to_filename(self, dataset, index_subset: int):
        fname = self.filenames_subset[index_subset]
        return fname

    @property
    def input_dir(self):
        return self.base_dataset.input_dir

    @property
    def dataset(self):
        return self.base_dataset.dataset
