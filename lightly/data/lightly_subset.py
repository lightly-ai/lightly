from typing import Dict, List, Optional, Tuple, Union

from lightly.data.dataset import LightlyDataset


class LightlySubset(LightlyDataset):  # type: ignore
    def __init__(self, base_dataset: LightlyDataset, filenames_subset: List[str]):
        """Creates a subset of a LightlyDataset by filtering samples based on their filenames.

        Args:
            base_dataset:
                The original dataset from which a subset will be created.
            filenames_subset:
                List of filenames to be included in the subset.
        """
        self.base_dataset = base_dataset
        self.filenames_subset = filenames_subset

        # Create a dictionary mapping filenames to their indices in the base dataset
        dict_base_dataset_filename_index: Dict[str, int] = {}
        for index in range(len(base_dataset)):
            fname = base_dataset.index_to_filename(base_dataset.dataset, index)
            dict_base_dataset_filename_index[fname] = index

        self.mapping_subset_index_to_baseset_index = [
            dict_base_dataset_filename_index[filename] for filename in filenames_subset
        ]

    def __getitem__(self, index_subset: int) -> Tuple[object, object, str]:
        """Retrieves a specific sample from the subset by its index.

        Args:
            index_subset:
                The index of a sample with respect to the subset.
                Index 0 corresponds to the first filename in filenames_subset.

        Returns:
            A tuple containing:
            - The sample data
            - The sample's target/label
            - The sample's filename
        """
        index_baseset = self.mapping_subset_index_to_baseset_index[index_subset]
        sample, target, fname = self.base_dataset.__getitem__(index_baseset)
        return sample, target, fname

    def __len__(self) -> int:
        """Returns the number of samples in the subset.

        Returns:
            The total count of samples in this subset.
        """
        return len(self.filenames_subset)

    def get_filenames(self) -> List[str]:
        """Retrieves the list of filenames in this dataset subset.

        Returns:
            A list of filenames included in this subset.
        """
        return self.filenames_subset

    def index_to_filename(
        self, dataset: Optional[Union[LightlyDataset, None]], index_subset: int
    ) -> str:
        """Converts a subset index to its corresponding filename.

        Args:
            dataset:
                Unused parameter to match the parent class method signature.
            index_subset:
                The index of the sample within the subset.

        Returns:
            The filename of the sample at the specified subset index.
        """
        fname = self.filenames_subset[index_subset]
        return fname

    @property
    def input_dir(self) -> str:
        """Provides access to the input directory of the base dataset.

        Returns:
            The input directory path from the original dataset.
        """
        return str(self.base_dataset.input_dir)

    @property
    def dataset(self) -> LightlyDataset:  # type: ignore
        """Provides access to the underlying dataset of the base dataset.

        Returns:
            The original dataset object.
        """
        return self.base_dataset.dataset
