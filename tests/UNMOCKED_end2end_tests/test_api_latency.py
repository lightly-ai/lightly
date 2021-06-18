import os
import time

import numpy as np
from tqdm import tqdm

from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import Configuration, ApiClient, QuotaApi

if __name__ == "__main__":
    token = os.getenv("TOKEN")
    api_client = ApiWorkflowClient(token=token).api_client
    quota_api = QuotaApi(api_client)

    no_iters = 200

    latencies = np.zeros(no_iters)
    for i in tqdm(range(no_iters)):
        start = time.time()
        quota_api.get_quota_maximum_dataset_size()
        duration = time.time()-start
        latencies[i] = duration

    def format_latency(latency: float):
        return f"{latency*1000:.1f}ms"

    values = [('min')]
    print(f"Latencies: min: {format_latency(np.min(latencies))}, mean: {format_latency(np.mean(latencies))}, max: {format_latency(np.max(latencies))}")
    print(f"\nPINGING TO GOOGLE")
    response = os.system("ping -c 1 " + "google.com")