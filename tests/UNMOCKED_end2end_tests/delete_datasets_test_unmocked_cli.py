import sys

from lightly.api import ApiWorkflowClient

if __name__ == "__main__":
    if len(sys.argv) == 1 + 3:
        num_datasets, token, date_time = \
            (sys.argv[1 + i] for i in range(3))
    else:
        raise ValueError("ERROR in number of command line arguments, must be 3."
                         "Example: python delete_datasets_test_unmocked_cli.py 6 LIGHTLY_TOKEN 2022-09-29-13-41-24")

    api_workflow_client = ApiWorkflowClient(token=token)

    num_datasets = int(num_datasets)
    for i in range(1, num_datasets+1):
        dataset_name = f"test_unmocked_cli_{i}_{date_time}"
        api_workflow_client.set_dataset_id_by_name(dataset_name)
        api_workflow_client.delete_dataset_by_id(api_workflow_client.dataset_id)
