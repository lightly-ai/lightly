from __future__ import annotations
from typing import TYPE_CHECKING, Union

from lightly.api.upload import upload_dataset
from lightly.api.upload import upload_images_from_folder

from lightly.data.dataset import LightlyDataset

if TYPE_CHECKING:
    from lightly.api.api_workflow_client import ApiWorkflowClient

import csv
from typing import List


class _UploadDatasetMixin:
    def upload_dataset(self: ApiWorkflowClient, input: Union[str, LightlyDataset], **kwargs):
        """Uploads a dataset to the server and creates the initial tag.

        Args:
            input:
                one of the following:
                    - the path to the dataset, e.g. "path/to/dataset"
                    - the dataset in form of a LightlyDataset
            **kwargs:
                see specification of the called functions
        """
        if isinstance(input, str):
            path_to_dataset = input
            upload_images_from_folder(path_to_dataset, self.dataset_id, self.token, **kwargs)
        elif isinstance(input, LightlyDataset):
            dataset = input
            upload_dataset(dataset, self.dataset_id, self.token, **kwargs)
        else:
            raise ValueError(f"input must either be a LightlyDataset or the path to the dataset as str, "
                             f"but is of type {type(input)}")
