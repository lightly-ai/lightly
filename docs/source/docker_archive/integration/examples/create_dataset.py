import lightly

# Create the Lightly client to connect to the API.
client = lightly.api.ApiWorkflowClient(token="LIGHTLY_TOKEN")

# Create a new dataset on the Lightly Platform.
client.create_dataset('dataset-name')

# Connect the dataset to your cloud bucket.

# AWS S3
client.set_s3_config(
    resource_path="s3://bucket/dataset/",
    region="eu-central-1",
    access_key="ACCESS-KEY",
    secret_access_key="SECRET",
    thumbnail_suffix=None,
)

# Google Cloud Storage
import json
client.set_gcs_config(
    resource_path="gs://bucket/dataset/",
    project_id="PROJECT-ID",
    credentials=json.dumps(json.load(open('credentials.json'))),
    thumbnail_suffix=None,
)

# Azure Blob Storage
client.set_azure_config(
    container_name="container/dataset/",
    account_name="ACCOUNT-NAME",
    sas_token="SAS-TOKEN",
    thumbnail_suffix=None,
)