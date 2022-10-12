import json
import lightly
from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType
from lightly.openapi_generated.swagger_client.models.datasource_purpose import DatasourcePurpose


# Create the Lightly client to connect to the API.
client = lightly.api.ApiWorkflowClient(token="YOUR_TOKEN")

# Create a new dataset on the Lightly Platform.
client.create_dataset('pedestrian-videos-datapool',
                      dataset_type=DatasetType.VIDEOS)

# Google Cloud Storage
# Input bucket
client.set_gcs_config(
    resource_path="gs://bucket/input/",
    project_id="PROJECT-ID",
    credentials=json.dumps(json.load(open('credentials_read.json'))),
    purpose=DatasourcePurpose.INPUT
)
# Output bucket
client.set_gcs_config(
    resource_path="gs://bucket/output/",
    project_id="PROJECT-ID",
    credentials=json.dumps(json.load(open('credentials_write.json'))),
    purpose=DatasourcePurpose.LIGHTLY
)

