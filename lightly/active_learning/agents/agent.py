import os
from typing import *
import time

import numpy as np

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.scorer import Scorer
from lightly.api.api_workflow import ApiWorkflow

from lightly.api.upload import upload_csv
from lightly.api.active_learning import upload_scores_to_api, sampling_request_to_api, get_job_status_from_api
from lightly.api.utils import create_api_client
from lightly.api.tags import get_tag_by_tag_id
from lightly.openapi_generated.swagger_client import JobStatusData, JobState
from lightly.api.bitmask import BitMask


class ActiveLearningAgent:
    def __init__(self, token: str = '', dataset_id: str = '', initial_tag: str = 'initial_tag',
                 path_to_embeddings: str = '', host: str = 'https://api-dev.lightly.ai'):

        os.environ['LIGHTLY_SERVER_LOCATION'] = host
        embedding_id = upload_csv(path_to_embeddings, dataset_id, token)

        self.api_workflow = ApiWorkflow(host=host, token=token, dataset_id=dataset_id, embedding_id=embedding_id)

    def sample(self, sampler_config: SamplerConfig, al_scorer: Scorer = None, preselected_tag_id: str = None) -> List[int]:

        # calculate scores
        if al_scorer is not None:
            scores = al_scorer._calculate_scores()
        n_samples = list(scores.values())[0].__len__()

        # return directly if this function should be mocked to work without the api
        use_mock = False
        if use_mock:
            return np.random.randint(0, n_samples, sampler_config.batch_size)

        # upload the scores
        if al_scorer is not None:
            upload_scores_to_api(self.api_client, scores)

        job_id = sampling_request_to_api(api_client=self.api_client, dataset_id=self.dataset_id,
                                         embedding_id=self.embedding_id, sampler_config=sampler_config,
                                         preselected_tag_id=preselected_tag_id, query_tag_id=self.current_tag)

        time.sleep(2)
        print(f"jobId: {job_id}")
        while True:

            job_status_data = get_job_status_from_api(self.api_client, job_id)
            print(job_status_data)
            if job_status_data.status == JobState.FINISHED:
                self.current_tag = job_status_data.result.data
                break
            elif job_status_data.status in [JobState.UNKNOWN, JobState.RUNNING, JobState.WAITING]:
                pass
            elif job_status_data.status == JobState.FAILED:
                raise ValueError(f"Sampling job failed with error {job_status_data.error}")

        # extract the chosen samples as List[int]
        chosen_samples = BitMask.from_bin(new_tag_data.bit_mask_data).to_indices()

        new_tag_data = get_tag_by_tag_id(api_client=self.api_client, dataset_id=self.dataset_id, tag_id=self.current_tag)
        chosen_samples = BitMask.from_bin(new_tag_data.bit_mask_data).to_indices()
        return chosen_samples
