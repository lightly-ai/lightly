''' Distributed Utils '''

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Any, Callable, Optional, Tuple, TypeVar

import torch
import torch.distributed as dist
from torch.autograd.function import FunctionCtx


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes, supporting backward propagation.

    This autograd function gathers input tensors from all processes in a distributed
    setup and supports backpropagation of gradients across processes.

    This code was adapted from:
    https://github.com/vturrisi/solo-learn/blob/b69b4bd27472593919956d9ac58902a301537a4d/solo/utils/misc.py#L187

    Methods:
        forward:
            Gathers the input tensor from all processes.
        backward:
            Aggregates the gradients from all processes and returns the gradient
            for the current process.

    """

    @staticmethod
    def forward(ctx: FunctionCtx, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass: Gathers the input tensor from all processes.

        Args:
            input:
                A tensor to be gathered from all processes.

        Returns:
            A tuple containing the gathered tensors from all processes.
        """
        output = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx: FunctionCtx, *grads: torch.Tensor) -> torch.Tensor:
        """Backward pass: Gathers gradients from all processes.

        Args:
            grads:
                Gradients from the output of the forward pass for each process.

        Returns:
            The gradient for the current process.
        """
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        grad_out = all_gradients[dist.get_rank()]
        return grad_out


def rank() -> int:
    """Returns the rank of the current process.

    Returns:
        The rank of the current process. If the process is not part of a distributed group,
        rank 0 is returned.
    """
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    """Returns the total number of distributed processes (world size).

    Returns:
        The total number of processes in the distributed group. If distributed training
        is not initialized, returns 1.
    """
    return dist.get_world_size() if dist.is_initialized() else 1


def gather(input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Gathers the input tensor from all processes.

    Args:
        input:
            A tensor to be gathered across all processes.

    Returns:
        A tuple of tensors, one for each process in the distributed group.
        Supports backpropagation.
    """
    return GatherLayer.apply(input)


def eye_rank(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Returns an (n, n * world_size) zero matrix with the diagonal for the current process rank set to 1.

    For undistributed settings or if the world size equals 1, this is equivalent to `torch.eye`.
    The diagonal corresponding to the current process's rank is set to 1, while all other entries remain 0.

    Args:
        n:
            The size of the square matrix for a single process.
        device:
            The device on which the tensor should be created.

    Returns:
        A tensor where the diagonal corresponding to the current process rank is set to 1.

    Example:
        For `n=3`, rank 1, and world size 4:

        rank0   rank1   rank2   rank3
        0 0 0 | 1 0 0 | 0 0 0 | 0 0 0
        0 0 0 | 0 1 0 | 0 0 0 | 0 0 0
        0 0 0 | 0 0 1 | 0 0 0 | 0 0 0
    """
    rows = torch.arange(n, device=device, dtype=torch.long)
    cols = rows + rank() * n
    diag_mask = torch.zeros((n, n * world_size()), dtype=torch.bool)
    diag_mask[(rows, cols)] = True
    return diag_mask


R = TypeVar("R")


def rank_zero_only(fn: Callable[..., R]) -> Callable[..., Optional[R]]:
    """Decorator that only executes a function on the process with rank 0.

    This is useful to ensure that certain functions (e.g., logging or printing) 
    only run once in distributed setups where multiple processes are involved.

    Args:
        fn:
            The function to be executed only on rank 0.

    Returns:
        A wrapped function that only executes on rank 0. For other ranks, it returns None.

    Example:
        >>> @rank_zero_only
        >>> def print_rank_zero(message: str):
        >>>     print(message)

        >>> print_rank_zero("Hello from rank 0!")  # Only prints from rank 0.
    """

    def wrapped(*args: Any, **kwargs: Any) -> Optional[R]:
        if rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped


@rank_zero_only
def print_rank_zero(*args: Any, **kwargs: Any) -> None:
    """Prints messages only on the process with rank 0.

    Args:
        *args:
            Arguments to be printed.
        **kwargs:
            Keyword arguments to be passed to the print function.

    Example:
        >>> print_rank_zero("This message is only printed by rank 0 process.")
    """
    print(*args, **kwargs)
