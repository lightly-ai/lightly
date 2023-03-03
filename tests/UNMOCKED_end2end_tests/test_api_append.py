import os
import sys

import torchvision

from lightly.data import LightlyDataset
from lightly.utils.io import format_custom_metadata
from tests.UNMOCKED_end2end_tests.test_api import \
    create_new_dataset_with_embeddings


def t_est_api_append(path_to_dataset: str, token: str,
                     dataset_name: str = "test_api_from_pip_append"):
    files_to_delete = []
    try:
        print("Save custom metadata")
        dataset = LightlyDataset(path_to_dataset)
        path_custom_metadata = f"{path_to_dataset}/custom_metadata.csv"
        custom_metadata = [(filename, {"metadata": f"{filename}_meta"}) for
            filename in dataset.get_filenames()]

        print("Upload to the dataset")
        api_workflow_client = create_new_dataset_with_embeddings(
            path_to_dataset=path_to_dataset, token=token,
            dataset_name=dataset_name)
        api_workflow_client.upload_custom_metadata(
            format_custom_metadata(custom_metadata))

        print("save additional images and embeddings and custom metadata")
        n_data = 5
        dataset = torchvision.datasets.FakeData(size=n_data,
                                                image_size=(3, 32, 32))
        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        for sample_idx in range(n_data):
            data = dataset[sample_idx]
            path = os.path.join(path_to_dataset, sample_names[sample_idx])
            files_to_delete.append(path)
            data[0].save(path)
        custom_metadata += [(filename, {"metadata": f"{filename}_meta"}) for
            filename in sample_names]

        print("Upload to the dataset")
        api_workflow_client.upload_dataset(path_to_dataset)

        print("Upload custom metadata")
        api_workflow_client.upload_custom_metadata(
            format_custom_metadata(custom_metadata))



    finally:
        for filename in files_to_delete:
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    if len(sys.argv) == 1 + 2:
        path_to_dataset, token = (sys.argv[1 + i] for i in range(2))
    else:
        raise ValueError(
            "ERROR in number of command line arguments, must be 2."
            "Example: python test_api path/to/dataset LIGHTLY_TOKEN")

    t_est_api_append(path_to_dataset=path_to_dataset, token=token)