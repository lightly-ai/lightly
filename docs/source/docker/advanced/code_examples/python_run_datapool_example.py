import json
import lightly
from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType

# Create the Lightly client to connect to the API.
client = lightly.api.ApiWorkflowClient(token="YOUR_TOKEN")

# Create a new dataset on the Lightly Platform.
client.create_dataset('pedestrian-videos-datapool', dataset_type=DatasetType.VIDEOS)

# Connect to Google Cloud Storage
client.set_gcs_config(
    resource_path="gs://path-to-your-bucket/my-folder/",
    project_id="your-project-name-1234",
    credentials=json.dumps(json.load(open('your-gcp-credentials.json'))),
    thumbnail_suffix=None,
)

# Schedule the compute run using our custom config.
# We show here the full default config so you can easily edit the
# values according to your needs.
client.schedule_compute_worker_run(
    worker_config={
        'enable_corruptness_check': True,
        'remove_exact_duplicates': True,
        'enable_training': False,
        'pretagging': True,
        'pretagging_debug': True,
        'method': 'coreset',
        'stopping_condition': {
            'n_samples': -1,
            'min_distance': 0.05 # we set the min_distance to 0.05 in this example
        },
        'scorer': 'object-frequency',
        'scorer_config': {
            'frequency_penalty': 0.25,
            'min_score': 0.9
        }
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

