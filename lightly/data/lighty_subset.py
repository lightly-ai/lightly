from typing import List, Dict, Tuple

from lightly.data.dataset import LightlyDataset


class LightlySubset(LightlyDataset):
    def __init__(self, base_dataset: LightlyDataset, filenames_subset: List[str]):
        self.base_dataset = base_dataset

        dict_base_dataset_filename_index: Dict[str, int] = \
            dict([
                (base_dataset.index_to_filename(self.dataset, index), index)
                for index in range(len(base_dataset))
            ])
        self.mapping_subset_index_baseset = \
            [dict_base_dataset_filename_index[filename] for filename in filenames_subset]

    def __getitem__(self, index_subset: int) -> Tuple[object, object, str]:
        index_baseset = self.mapping_subset_index_baseset[index_subset]
        sample, target, fname = self.base_dataset.__getitem__(index_baseset)
        return sample, target, fname

    def __len__(self) -> int:
        return len(self.mapping_subset_index_baseset)

    def index_to_filename(self, dataset, index_subset: int):
        index_baseset = self.mapping_subset_index_baseset[index_subset]
        fname = self.base_dataset.index_to_filename(dataset, index_baseset)
        return fname

    @property
    def input_dir(self):
        return self.base_dataset.input_dir

    @property
    def dataset(self):
        return self.base_dataset.dataset
