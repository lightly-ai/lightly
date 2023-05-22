import json

import lightly
from lightly.openapi_client.models.dataset_type import DatasetType
from lightly.openapi_client.models.datasource_purpose import DatasourcePurpose

# Create the Lightly client to connect to the API.
client = lightly.api.ApiWorkflowClient(token="YOUR_TOKEN")

# Create a new dataset on the Lightly Platform.
client.create_dataset("pedestrian-videos-datapool", dataset_type=DatasetType.VIDEOS)

# Azure Blob Storage
# Input bucket
client.set_azure_config(
    container_name="my-container/input/",
    account_name="ACCOUNT-NAME",
    sas_token="SAS-TOKEN",
    purpose=DatasourcePurpose.INPUT,
)
# Output bucket
client.set_azure_config(
    container_name="my-container/output/",
    account_name="ACCOUNT-NAME",
    sas_token="SAS-TOKEN",
    purpose=DatasourcePurpose.LIGHTLY,
)
