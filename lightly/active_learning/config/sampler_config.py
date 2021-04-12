from datetime import datetime

from lightly.openapi_generated.swagger_client.models.sampling_method import SamplingMethod


class SamplerConfig:
    """Configuration class for a sampler.

    Attributes:
        method:
            The method to use for sampling, one of CORESET, RANDOM, CORAL, ACTIVE_LEARNING
        n_samples:
            The maximum number of samples to be chosen by the sampler
            including the samples in the preselected tag. One of the stopping
            conditions.
        min_distance:
            The minimum distance of samples in the chosen set, one of the
            stopping conditions.
        name:
            The name of this sampling, defaults to a name consisting of all
            other attributes and the datetime. A new tag will be created in the
            web-app under this name.

    Examples:
        >>> # sample 100 images with CORESET sampling
        >>> config = SamplerConfig(method=SamplingMethod.CORESET, n_samples=100)
        >>> config = SamplerConfig(method=SamplingMethod.CORESET, n_samples=100)
        >>>
        >>> # give your sampling a name
        >>> config = SamplerConfig(method=SamplingMethod.CORESET, n_samples=100, name='my-sampling')
        >>>
        >>> # use minimum distance between samples as stopping criterion
        >>> config = SamplerConfig(method=SamplingMethod.CORESET, n_samples=-1, min_distance=0.1)

    """
    def __init__(self, method: SamplingMethod = SamplingMethod.CORESET, n_samples: int = 32, min_distance: float = -1,
                 name: str = None):

        self.method = method
        self.n_samples = n_samples
        self.min_distance = min_distance
        if name is None:
            date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
            name = f"{self.method}_{self.n_samples}_{self.min_distance}_{date_time}"
        self.name = name

