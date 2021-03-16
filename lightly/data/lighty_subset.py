from typing import List, Dict, Tuple

from lightly.data.dataset import LightlyDataset


class LightlySubset(LightlyDataset):
    def __init__(self, base_dataset: LightlyDataset, filenames_subset: List[str]):
        """Creates a subset of a LightlyDataset.

        Args:
            base_dataset:
                The dataset to subset from.
            filenames_subset:
                The filenames of the samples to be part of the subset.
        """
        self.base_dataset = base_dataset
        self.filenames_subset = filenames_subset

        dict_base_dataset_filename_index: Dict[str, int] = dict()
        for index in range(len(base_dataset)):
            fname = base_dataset.index_to_filename(self.dataset, index)
            dict_base_dataset_filename_index[fname] = index

        self.mapping_subset_index_to_baseset_index = \
            [dict_base_dataset_filename_index[filename] for filename in filenames_subset]

    def __getitem__(self, index_subset: int) -> Tuple[object, object, str]:
        """An overwrite for indexing.

        Args:
            index_subset:
                The index of a sample w.r.t. to the subset.
                E.g. if index_subset == 0, the sample belonging to
                the first filename in self.filenames_subset is returned.

        Returns:
            A tuple of the sample, its target and its filename.

        """
        index_baseset = self.mapping_subset_index_to_baseset_index[index_subset]
        sample, target, fname = self.base_dataset.__getitem__(index_baseset)
        return sample, target, fname

    def __len__(self) -> int:
        """Overwrites the len(...) function.

        Returns:
            The number of samples in the subset.
        """
        return len(self.filenames_subset)

    def index_to_filename(self, dataset, index_subset: int):
        """Maps from an index of a sample to its filename.

        Args:
            dataset:
                Unused, but specified by the overwritten
                function of the parent class.
            index_subset:
                The index of the sample w.r.t. the subset.

        Returns:
            The filename of the sample.
        """
        fname = self.filenames_subset[index_subset]
        return fname

    @property
    def input_dir(self):
        return self.base_dataset.input_dir

    @property
    def dataset(self):
        return self.base_dataset.dataset
