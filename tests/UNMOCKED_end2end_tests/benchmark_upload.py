import sys

from hydra.experimental import initialize, compose

from lightly.api import ApiWorkflowClient
from lightly.cli import upload_cli

def benchmark_upload(path_to_dataset, token):
    api_workflow_client = ApiWorkflowClient(token=token)

    # create the dataset
    api_workflow_client.create_new_dataset_with_unique_name(dataset_basename="benchmark_upload")

    # upload to the dataset
    initialize(config_path="../../lightly/cli/config", job_name="test_app")
    cfg = compose(config_name="config", overrides=[
        f"input_dir='{path_to_dataset}'",
        f"token='{token}'",
        f"dataset_id={api_workflow_client.dataset_id}",
        f"loader.num_workers=12"
    ])
    upload_cli(cfg)
    api_workflow_client.delete_dataset_by_id(api_workflow_client.dataset_id)


if __name__ == "__main__":
    if len(sys.argv) == 1 + 2:
        path_to_dataset, token = \
            (sys.argv[1 + i] for i in range(2))
    else:
        raise ValueError("ERROR in number of command line arguments, must be 2."
                         "Example: python test_api path/to/dataset TOKEN")

    benchmark_upload(path_to_dataset=path_to_dataset, token=token)
