import torch
import numpy as np 


def cosine_schedule(
    step: int, max_steps: int, start_value: float, end_value: float
) -> float:

    """
    Use cosine decay to gradually modify start_value to reach target end_value during iterations.

    Args:
        step:
            Current step number.
        max_steps:
            Total number of steps.
        start_value:
            Starting value.
        end_value:
            Target value.

    Returns:
        Cosine decay value.

    """
    if step < 0:
        raise ValueError("Current step number can't be negative")
    if max_steps < 1:
        raise ValueError("Total step number must be >= 1")
    if step >= max_steps:
        raise ValueError(
            f"The current step must be smaller than max_steps but found step equal to {step} and max_steps equal to {max_steps}."
        )

    if max_steps == 1:
        decay = end_value
    else:
        decay = (
            end_value
            - (end_value - start_value)
            * (np.cos(np.pi * step / (max_steps - 1)) + 1)
            / 2
        )
    return decay


class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Cosine warmup scheduler for learning rate.

    Args:
        optimizer:
            Optimizer object to schedule the learning rate.
        warmup_epochs:
            Number of warmup epochs.
        max_epochs:
            Total number of training epochs.
        last_epoch:
            The index of last epoch. Default: -1
        verbose:
            If True, prints a message to stdout for each update. Default: False.
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer=optimizer, lr_lambda=self.scale_lr, last_epoch=last_epoch, verbose=verbose)

    def scale_lr(self, epoch):
        """
        Scale learning rate according to the current epoch number.

        Args:
            epoch:
                Current epoch number.

        Returns:
            Scaled learning rate.

        """
        if epoch < self.warmup_epochs:
            return epoch / self.warmup_epochs
        else:
            return cosine_schedule(epoch - self.warmup_epochs, self.max_epochs - self.warmup_epochs, 0, 1)