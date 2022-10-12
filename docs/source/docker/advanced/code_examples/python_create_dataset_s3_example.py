import json
import lightly
from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType
from lightly.openapi_generated.swagger_client.models.datasource_purpose import DatasourcePurpose


# Create the Lightly client to connect to the API.
client = lightly.api.ApiWorkflowClient(token="YOUR_TOKEN")

# Create a new dataset on the Lightly Platform.
client.create_dataset('pedestrian-videos-datapool',
                      dataset_type=DatasetType.VIDEOS)

# AWS S3
# Input bucket
client.set_s3_config(
    resource_path="s3://bucket/input/",
    region='eu-central-1',
    access_key='S3-ACCESS-KEY',
    secret_access_key='S3-SECRET-ACCESS-KEY',
    purpose=DatasourcePurpose.INPUT
)
# Output bucket
client.set_s3_config(
    resource_path="s3://bucket/output/",
    region='eu-central-1',
    access_key='S3-ACCESS-KEY',
    secret_access_key='S3-SECRET-ACCESS-KEY',
    purpose=DatasourcePurpose.LIGHTLY
)

