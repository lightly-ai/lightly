# This file is only for backwards compatibility.
# Scheduler implementation has moved to lightly/schedulers and lightly/lr_schedulers.

from lightly.lr_schedulers.cosine_warmup_lr import (
    CosineWarmupLR as CosineWarmupScheduler,
)
from lightly.schedulers import cosine_schedule
