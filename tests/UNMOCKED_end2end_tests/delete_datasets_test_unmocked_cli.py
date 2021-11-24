import sys

from lightly.api import ApiWorkflowClient

if __name__ == "__main__":
    if len(sys.argv) == 1 + 2:
        num_datasets, token = \
            (sys.argv[1 + i] for i in range(2))
    else:
        raise ValueError("ERROR in number of command line arguments, must be 2."
                         "Example: python test_api 6 TOKEN")

    api_workflow_client = ApiWorkflowClient(token=token)

    num_datasets = int(num_datasets)
    for i in range(1, num_datasets+1):
        dataset_name = f"test_unmocked_cli_{i}"
        api_workflow_client.create_dataset(dataset_name)
        api_workflow_client.delete_dataset_by_id(api_workflow_client.dataset_id)
