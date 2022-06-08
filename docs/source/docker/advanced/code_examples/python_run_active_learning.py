import json
import lightly
from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType

# Create the Lightly client to connect to the API.
client = lightly.api.ApiWorkflowClient(token="YOUR_TOKEN")

# Create a new dataset on the Lightly Platform.
client.create_dataset('pedestrian-videos-datapool', dataset_type=DatasetType.VIDEOS)

# Pick one of the following three blocks depending on where your data is
# AWS S3
client.set_s3_config(
    resource_path="s3://bucket/dataset/",
    region="eu-central-1",
    access_key="ACCESS-KEY",
    secret_access_key="SECRET",
    thumbnail_suffix=None,
)

# or Google Cloud Storage
client.set_gcs_config(
    resource_path="gs://bucket/dataset/",
    project_id="PROJECT-ID",
    credentials=json.dumps(json.load(open('credentials.json'))),
    thumbnail_suffix=None,
)

# or Azure Blob Storage
client.set_azure_config(
    container_name="container/dataset/",
    account_name="ACCOUNT-NAME",
    sas_token="SAS-TOKEN",
    thumbnail_suffix=None,
)

# Schedule the docker run with 
#  - "active_learning.task_name" set to your task name
#  - "method" set to "coral"
# All other settings are default values and we show them so you can easily edit
# the values according to your need.
client.schedule_compute_worker_run(
    worker_config={
        "enable_corruptness_check": True,
        "remove_exact_duplicates": True,
        "enable_training": False,
        "pretagging": False,
        "pretagging_debug": False,
        "method": "coral",
        "stopping_condition": {
          "n_samples": 0.1,
          "min_distance": -1
        },
        "scorer": "object-frequency",
        "scorer_config": {
          "frequency_penalty": 0.25,
          "min_score": 0.9
        },
        "active_learning": { # here we specify our active learning parameters
          "task_name": "my-classification-task",    # set the task
          "score_name": "uncertainty_margin"        # set the score
        }
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