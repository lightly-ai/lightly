import warnings
from datetime import datetime

from lightly.openapi_generated.swagger_client.models.sampling_method import SamplingMethod


class SamplingConfig(SelectionConfig):

    def __init__(self, *args, **kwargs):
        warnings.warn(PendingDeprecationWarning(
            "SamplingConfig() is deprecated "
            "in favour of SelectionConfig() "
            "and will be removed in the future."
        ), )
        SelectionConfig(self).__init__(*args, **kwargs)



class SelectionConfig:
    """Configuration class for a selection.

    Attributes:
        method:
            The method to use for selection, one of CORESET, RANDOM, CORAL, ACTIVE_LEARNING
        n_samples:
            The maximum number of samples to be chosen by the selection
            including the samples in the preselected tag. One of the stopping
            conditions.
        min_distance:
            The minimum distance of samples in the chosen set, one of the
            stopping conditions.
        name:
            The name of this selection, defaults to a name consisting of all
            other attributes and the datetime. A new tag will be created in the
            web-app under this name.

    Examples:
        >>> # select 100 images with CORESET selection
        >>> config = SelectionConfig(method=SamplingMethod.CORESET, n_samples=100)
        >>>
        >>> # give your selection a name
        >>> config = SelectionConfig(method=SamplingMethod.CORESET, n_samples=100, name='my-selection')
        >>>
        >>> # use minimum distance between samples as stopping criterion
        >>> config = SelectionConfig(method=SamplingMethod.CORESET, n_samples=-1, min_distance=0.1)

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

