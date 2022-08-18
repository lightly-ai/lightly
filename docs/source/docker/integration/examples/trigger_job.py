# You can reuse the client previous scripts. If you want to create a new
# one you can uncomment the following line:
# client = lightly.api.ApiWorkflowClient(token="TOKEN", dataset_id="DATASET_ID")

# Schedule the compute run using a custom config.
# You can easily edit the values according to your needs.


client.schedule_compute_worker_run(
    worker_config={
        'enable_corruptness_check': True,
        'remove_exact_duplicates': True,
        'enable_training': False,
    },
    selection_config={
        "nSamples": 50,
        "strategies": [
            {
                "input": {
                    "type": "EMBEDDINGS"
                },
                "strategy": {
                    "type": "DIVERSIFY"
                }
            }
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
