import json
import lightly
from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType
from lightly.openapi_generated.swagger_client.models.datasource_purpose import DatasourcePurpose

# Create the Lightly client to connect to the API.
client = lightly.api.ApiWorkflowClient(token="YOUR_TOKEN")

# Create a new dataset on the Lightly Platform.
client.create_dataset('pexels', dataset_type=DatasetType.VIDEOS)

# Pick one of the following three blocks depending on where your data is
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


# or Google Cloud Storage
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


# or Azure Blob Storage
# Input bucket
client.set_azure_config(
    container_name='my-container/input/',
    account_name='ACCOUNT-NAME',
    sas_token='SAS-TOKEN',
    purpose=DatasourcePurpose.INPUT
)
# Output bucket
client.set_azure_config(
    container_name='my-container/output/',
    account_name='ACCOUNT-NAME',
    sas_token='SAS-TOKEN',
    purpose=DatasourcePurpose.LIGHTLY
)

# Schedule the compute run using our custom config.
# We show here the full default config so you can easily edit the
# values according to your needs.
client.schedule_compute_worker_run(
    worker_config={
        'enable_corruptness_check': False,
        'remove_exact_duplicates': False,
        'enable_training': False,
        'pretagging': False,
        'pretagging_debug': False,
        'method': 'coreset',
        'stopping_condition': {
            'n_samples': 200, # select 200 frames of length 10 frames -> 20 sequences
            'min_distance': -1
        },
        'selected_sequence_length': 10 # we want sequences of 10 frames lenght
    },
    lightly_config={
        'loader': {
            'batch_size': 128,
            'shuffle': True,
            'num_workers': -1,
            'drop_last': True
        },
        'model': {
            'name': 'resnet-18',
            'out_dim': 128,
            'num_ftrs': 32,
            'width': 1
        },
        'trainer': {
            'gpus': 1,
            'max_epochs': 1,
            'precision': 16
        },
        'criterion': {
            'temperature': 0.5
        },
        'optimizer': {
            'lr': 1,
            'weight_decay': 0.00001
        },
        'collate': {
            'input_size': 64,
            'cj_prob': 0.8,
            'cj_bright': 0.7,
            'cj_contrast': 0.7,
            'cj_sat': 0.7,
            'cj_hue': 0.2,
            'min_scale': 0.15,
            'random_gray_scale': 0.2,
            'gaussian_blur': 0.0,
            'kernel_size': 0.1,
            'vf_prob': 0,
            'hf_prob': 0.5,
            'rr_prob': 0
        }
    }
)

