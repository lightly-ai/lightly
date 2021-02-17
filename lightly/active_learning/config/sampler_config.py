from datetime import datetime

from lightly.openapi_generated.swagger_client.models.sampling_method import SamplingMethod


class SamplerConfig:
    """The configuration parameters of a sampler

    This class should be created by the user.

    Attributes:
        method:
            The method to use for sampling, e.g. RANDOM.
        batch_size:
            The maximum number of samples to be chosen by the sampler.
        min_distance:
            The minimum distance of samples in the chosen set, one of stopping conditions.
        name:
            The name of this sampling, defaults to a name consisting of all other attributes and the datetime

    """
    def __init__(self, method: SamplingMethod = SamplingMethod.RANDOM, batch_size: int = 32, min_distance: float = -1,
                 name: str = None):

        self.method = method
        self.batch_size = batch_size
        self.min_distance = min_distance
        if name is None:
            date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
            name = f"{self.method}_{self.batch_size}_{self.min_distance}_{date_time}"
        self.name = name

