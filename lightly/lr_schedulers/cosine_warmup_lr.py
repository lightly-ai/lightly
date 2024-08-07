from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from lightly.schedulers import cosine_scheduler


class CosineWarmupLR(LambdaLR):
    """Cosine warmup scheduler for learning rate.

    Args:
        optimizer:
            Optimizer object to schedule the learning rate.
        warmup_epochs:
            Number of warmup epochs or steps.
        max_epochs:
            Total number of training epochs or steps.
        last_epoch:
            The index of last epoch or step. Default: -1
        start_value:
            Starting learning rate scale. Default: 1.0
        end_value:
            Target learning rate scale. Default: 0.001
        verbose:
            If True, prints a message to stdout for each update. Default: False.

    Note: The `epoch` arguments do not necessarily have to be epochs. Any step or index
    can be used. The naming follows the Pytorch convention to use `epoch` for the steps
    in the scheduler.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        last_epoch: int = -1,
        start_value: float = 1.0,
        end_value: float = 0.001,
        period: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.start_value = start_value
        self.end_value = end_value
        self.period = period
        super().__init__(
            optimizer=optimizer,
            lr_lambda=self.scale_lr,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def scale_lr(self, epoch: int) -> float:
        """
        Scale learning rate according to the current epoch number.

        Args:
            epoch:
                Current epoch number.

        Returns:
            Scaled learning rate.

        """
        return cosine_scheduler.cosine_warmup_schedule(
            step=epoch,
            max_steps=self.max_epochs,
            start_value=self.start_value,
            end_value=self.end_value,
            warmup_steps=self.warmup_epochs,
            warmup_start_value=0.0,
            warmup_end_value=self.start_value,
            period=self.period,
        )
