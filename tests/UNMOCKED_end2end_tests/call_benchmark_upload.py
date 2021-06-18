import os

from tests.UNMOCKED_end2end_tests.benchmark_upload import benchmark_upload

path_to_dataset = "/path/to/my/dataset"
token = os.environ[f"TOKEN"]

benchmark_upload(path_to_dataset=path_to_dataset, token=token)