from datetime import datetime

from lightly.openapi_generated_with_other_gen.openapi_client.model.sampling_method import SamplingMethod
from lightly.openapi_generated_with_other_gen.openapi_client.model.sampling_config import SamplingConfig
from lightly.openapi_generated_with_other_gen.openapi_client.model.sampling_create_request import SamplingCreateRequest
from lightly.openapi_generated_with_other_gen.openapi_client.model.sampling_config_stopping_condition import \
    SamplingConfigStoppingCondition


class SamplerConfig:
    def __init__(self, method: SamplingMethod = SamplingMethod(value="RANDOM"), batch_size: int = 32, min_distance: float = -1,
                 name: str = None):
        self.method = method
        self.batch_size = batch_size
        self.min_distance = min_distance
        if name is None:
            date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            name = f"{self.method}_{self.batch_size}_{self.min_distance}_{date_time}"
        self.name = name

    def get_as_api_sampling_create_request(self, preselected_tag_id: str = None,
                                           query_tag_id: str = None) -> SamplingCreateRequest:
        sampling_config = SamplingConfig(
            stopping_condition=SamplingConfigStoppingCondition(self.batch_size, self.min_distance))
        sampling_create_request = SamplingCreateRequest(name=self.name, method=self.method, config=sampling_config,
                                                        preselected_tag_id=preselected_tag_id,
                                                        query_tag_id=query_tag_id)
        return sampling_create_request
