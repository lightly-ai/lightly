import sys
import time

from hydra.experimental import initialize, compose

from lightly.api import ApiWorkflowClient
from lightly.cli import upload_cli

def benchmark_upload(path_to_dataset, token, num_workers: int=-1):
    """
    This function creates a new dataset on the Lightly web platform.
    Then it uploads the dataset to it using the function for the cli command.
    Last, it deletes the dataset again.

    Call this function with a profiler to see which functions take up the most time
    and how long the upload takes in total. This knowledge allows to optimize the upload.

    Args:
        path_to_dataset:
            Filepath to the dataset to be uploaded.
        token:
            The token of the Lightly Web platform
        num_workers:
            The number of workers uploading in parallel


    """
    start = time.time()

    api_workflow_client = ApiWorkflowClient(token=token)

    # create the dataset
    api_workflow_client.create_new_dataset_with_unique_name(dataset_basename="benchmark_upload")

    # upload to the dataset
    initialize(config_path="../../lightly/cli/config", job_name="test_app")
    cfg = compose(config_name="config", overrides=[
        f"input_dir='{path_to_dataset}'",
        f"token='{token}'",
        f"dataset_id={api_workflow_client.dataset_id}",
        f"loader.num_workers={num_workers}"
    ])
    upload_cli(cfg)

    #delete the dataset again
    api_workflow_client.delete_dataset_by_id(api_workflow_client.dataset_id)

    duration = time.time() - start
    print(f"Finished and need {duration:.3f}s in total.")


if __name__ == "__main__":
    if len(sys.argv) == 1 + 2:
        path_to_dataset, token = \
            (sys.argv[1 + i] for i in range(2))
    else:
        raise ValueError("ERROR in number of command line arguments, must be 2."
                         "Example: python benchmark_upload path/to/dataset TOKEN")

    benchmark_upload(path_to_dataset=path_to_dataset, token=token)
