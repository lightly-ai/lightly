from lightly.api import ApiWorkflowClient


client = ApiWorkflowClient(token='347f1dfbc3879a142d536d0b', dataset_id='62987e77565fcdbf75c76268')


client.schedule_compute_worker_run(
    worker_config={
        "relevant_filenames_file": "relevant_filenames.txt",
        "enable_corruptness_check": True,
        "remove_exact_duplicates": True,
        "enable_training": False,
        "pretagging": False,
        "pretagging_debug": False,
        "method": "coreset",
        "stopping_condition": {
            "n_samples": 0.1,
            "min_distance": -1
        },
        "scorer": "object-frequency",
        "scorer_config": {
            "frequency_penalty": 0.25,
            "min_score": 0.9
        },
        "active_learning": {
            "task_name": "",
            "score_name": "uncertainty_margin"
        },
        "object_level": {
            "task_name": ""
        }
    }
)