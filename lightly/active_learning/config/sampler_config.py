from datetime import datetime

from lightly.openapi_generated.swagger_client.models.sampling_method import SamplingMethod
from lightly.openapi_generated.swagger_client.models.sampling_config import SamplingConfig
from lightly.openapi_generated.swagger_client.models.sampling_create_request import SamplingCreateRequest
from lightly.openapi_generated.swagger_client.models.sampling_config_stopping_condition import \
    SamplingConfigStoppingCondition


class SamplerConfig:
    """The configuration parameters of a sampler

    Args:
        method: the method to use for sampling, e.g. RANDOM
        batch_size: the number of samples to choose by the sampler, one of the stopping conditions
        min_distance: the minimum distance of samples in the chosen set, one of stopping conditions
        name: the name of this sampling, defaults to a name consisting of all other arguments and the datetime
    """
    def __init__(self, method: SamplingMethod = SamplingMethod.RANDOM, batch_size: int = 32, min_distance: float = -1,
                 name: str = None):

        self.method = method
        self.batch_size = batch_size
        self.min_distance = min_distance
        if name is None:
            date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            name = f"{self.method}_{self.batch_size}_{self.min_distance}_{date_time}"
        self.name = name

    def _get_as_api_sampling_create_request(self, preselected_tag_id: str = None,
                                            query_tag_id: str = None) -> SamplingCreateRequest:
        """Creates a sampling request as payload for the api.

        This method is used internally by lightly/api/api_workflow_sampling:sampling()

        Args:
            preselected_tag_id: The tag defining the already chosen samples (e.g. already labelled ones), default: None
            query_tag_id: The tag defining where to sample from, default: initial_tag

        Returns:
            the sampling request used as payload

        """
        sampling_config = SamplingConfig(
            stopping_condition=SamplingConfigStoppingCondition(self.batch_size, self.min_distance))
        sampling_create_request = SamplingCreateRequest(new_tag_name=self.name,
                                                        method=self.method,
                                                        config=sampling_config,
                                                        preselected_tag_id=preselected_tag_id,
                                                        query_tag_id=query_tag_id)
        return sampling_create_request
