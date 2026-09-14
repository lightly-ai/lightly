"""How a loss behaves when the batch is split across ranks."""

from enum import Enum, auto

__all__ = ["DistributedKind"]


class DistributedKind(Enum):
    """What a loss needs from the ranks around it.

    DDP averages the per-rank loss and the per-rank gradient. Whether that is
    correct depends on the loss, and the difference is not visible in a forward
    pass: issue #1920 shipped a correct SIGReg forward with a gradient off by
    exactly ``1 / world_size``, and came back as #1977 against
    ``BarlowTwinsLoss``. Declaring the kind is what selects the test.
    """

    RANK_LOCAL = auto()
    """The per-rank value is a valid estimate, so DDP averaging is correct.

    An MSE reconstruction loss.
    """

    GATHER_FOR_NEGATIVES = auto()
    """Correct only if the features are gathered with gradient first.

    NT-Xent and CLIP: every sample on every rank is a negative for every other.
    """

    GLOBAL_STATISTIC = auto()
    """A function of the global batch, so averaging per rank is wrong.

    SIGReg and the distribution-matching regularisers. The gap between the two
    is the variance across ranks, which closes only once the ranks agree.
    """
