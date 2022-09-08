import json
import lightly
from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType
from lightly.openapi_generated.swagger_client.models.datasource_purpose import DatasourcePurpose

# Create the Lightly client to connect to the API.
client = lightly.api.ApiWorkflowClient(token="YOUR_TOKEN")

# Create a new dataset on the Lightly Platform.
client.create_dataset('dataset-name',
                      dataset_type=DatasetType.IMAGES)

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

# Schedule the docker run with the "object_level.task_name" argument set. 
# All other settings are default values and we show them so you can easily edit
# the values according to your need.
client.schedule_compute_worker_run(
    worker_config={
        "enable_corruptness_check": True,
        "remove_exact_duplicates": True,
        "enable_training": False,
        "pretagging": False,
        "pretagging_debug": False,
        "object_level": { # used for object level workflow
            "task_name": "vehicles_object_detections" 
        },
    },
    selection_config={
        "n_samples": 100,
        "strategies": [
            {
                "input": {
                    "type": "EMBEDDINGS"
                },
                "strategy": {
                    "type": "DIVERSITY",
                }
            },
            # Optionally, you can combine diversity selection with active learning
            # to prefer selecting objects the model struggles with.
            # If you want that, just include the following code:
            """
            {
                "input": {
                    "type": "SCORES",
                    "task": "vehicles_object_detections", # change to your task
                    "score": "uncertainty_entropy" # change to your preferred score
                },
                "strategy": {
                    "type": "WEIGHTS"
                }
            }
            """
        ]
    },
    lightly_config={
        'loader': {
            'batch_size': 16,
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
            'max_epochs': 100,
            'precision': 32
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
            'gaussian_blur': 0.5,
            'kernel_size': 0.1,
            'vf_prob': 0,
            'hf_prob': 0.5,
            'rr_prob': 0
        }
    }
)
