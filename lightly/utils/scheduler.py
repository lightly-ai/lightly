import numpy as np
import torch


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
    if step > max_steps:
        # Note: we allow step == max_steps even though step starts at 0 and should end
        # at max_steps - 1. This is because Pytorch Lightning updates the LR scheduler
        # always for the next epoch, even after the last training epoch. This results in
        # Pytorch Lightning calling the scheduler with step == max_steps.
        raise ValueError(
            f"The current step cannot be larger than max_steps but found step {step} and max_steps {max_steps}."
        )

    if max_steps == 1:
        # Avoid division by zero
        decay = end_value
    elif step == max_steps:
        # Special case for Pytorch Lightning which updates LR scheduler also for epoch
        # after last training epoch.
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

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
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
        if epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs
        else:
            return cosine_schedule(
                step=epoch - self.warmup_epochs,
                max_steps=self.max_epochs - self.warmup_epochs,
                start_value=1.0,
                end_value=0.0,
            )
