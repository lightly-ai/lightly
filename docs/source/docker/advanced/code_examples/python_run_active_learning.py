import json
import lightly
from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType
from lightly.openapi_generated.swagger_client.models.datasource_purpose import DatasourcePurpose
from lightly.openapi_generated.swagger_client import DockerWorkerSelectionInputType, DockerWorkerSelectionStrategyType, DockerWorkerSelectionConfig, \
    DockerWorkerSelectionConfigEntry, DockerWorkerSelectionConfigEntryInput, DockerWorkerSelectionConfigEntryStrategy, \
    DockerWorkerSelectionInputPredictionsName

# Create the Lightly client to connect to the API.
client = lightly.api.ApiWorkflowClient(token="YOUR_TOKEN")

# Create a new dataset on the Lightly Platform.
client.create_dataset('pedestrian-videos-datapool',
                      dataset_type=DatasetType.VIDEOS)

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

# in this example we use a diversifying selection strategy (CORESET)
selection_config = DockerWorkerSelectionConfig(
    n_samples=100,
    strategies=[
        DockerWorkerSelectionConfigEntry(
            input=DockerWorkerSelectionConfigEntryInput(
                type=DockerWorkerSelectionInputType.EMBEDDINGS
            ),
            strategy=DockerWorkerSelectionConfigEntryStrategy(
                type=DockerWorkerSelectionStrategyType.DIVERSIFY
            )
        ),
        DockerWorkerSelectionConfigEntry(
            input=DockerWorkerSelectionConfigEntryInput(
                type=DockerWorkerSelectionInputType.SCORES,
                task="my_object_detection_task", # change to your task
                score="uncertainty_entropy" # change to your preferred score
            ),
            strategy=DockerWorkerSelectionConfigEntryStrategy(
                type=DockerWorkerSelectionStrategyType.WEIGHTS
            )
        ),
        DockerWorkerSelectionConfigEntry(
            input=DockerWorkerSelectionConfigEntryInput(
                type=DockerWorkerSelectionInputType.PREDICTIONS,
                task="my_object_detection_task", 
                name=DockerWorkerSelectionInputPredictionsName.CLASS_DISTRIBUTION
            ),
            strategy=DockerWorkerSelectionConfigEntryStrategy(
                type=DockerWorkerSelectionStrategyType.BALANCE,
                target={
                    "car": 0.1, # add your own classes here (defined in your `schema.json`)
                    "bicycle": 0.5, 
                    "bus": 0.1, 
                    "motorcycle": 0.1, 
                    "person": 0.1, 
                    "train": 0.05, 
                    "truck": 0.05
                }
            )
        )
    ]
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
    },
    selection_config=selection_config,
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
