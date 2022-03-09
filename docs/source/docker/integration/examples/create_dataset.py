import lightly

# we create the Lightly client to connect to the API
client = lightly.api.ApiWorkflowClient(token="TOKEN")

# (optonally) create a new dataset using Python code
# and connect it to an existing S3 bucket
client.create_dataset('dataset-name')
client.set_s3_config(
    resource_path="s3://bucket/dataset",
    region="eu-central-1",
    access_key="KEY",
    secret_access_key="SECRET",
    thumbnail_suffix=None,
)
