from __future__ import annotations
from typing import TYPE_CHECKING, Union

from lightly.api.upload import upload_dataset

from lightly.data.dataset import LightlyDataset

if TYPE_CHECKING:
    from lightly.api.api_workflow_client import ApiWorkflowClient

import csv
from typing import List


class _UploadDatasetMixin:
    def upload_dataset(self: ApiWorkflowClient, input: Union[str, LightlyDataset]):
        """Uploads a dataset to the server and creates the initial tag.

        Args:
            input:
                one of the following:
                    - the path to the dataset, e.g. "path/to/dataset"
                    - the dataset in form of a LightlyDataset
        """
        if isinstance(input, str):
            dataset = LightlyDataset(input_dir=input)
        elif isinstance(input, LightlyDataset):
            dataset = input
        else:
            raise ValueError(f"input must either be a LightlyDataset or the path to the dataset as str, "
                             f"but is of type {type(input)}")
        upload_dataset(dataset, self.dataset_id, self.token)
