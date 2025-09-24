from typing import Any, Callable, Optional, Tuple, TypeVar

import torch
import torch.distributed as dist
from torch.autograd.function import FunctionCtx


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes, supporting backward propagation.

    Adapted from the Solo-Learn project:
    https://github.com/vturrisi/solo-learn/blob/b69b4bd27472593919956d9ac58902a301537a4d/solo/utils/misc.py#L187

    """

    @staticmethod
    def forward(ctx: FunctionCtx, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:  # type: ignore
        output = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx: FunctionCtx, *grads: torch.Tensor) -> torch.Tensor:  # type: ignore
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        grad_out = all_gradients[dist.get_rank()]
        return grad_out


def rank() -> int:
    """Returns the rank of the current process."""
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    """Returns the current world size (number of distributed processes)."""
    return dist.get_world_size() if dist.is_initialized() else 1


def gather(input: torch.Tensor) -> Tuple[torch.Tensor]:
    """Gathers a tensor from all processes and supports backpropagation."""
    return GatherLayer.apply(input)  # type: ignore[no-any-return]


def eye_rank(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Returns an (n, n * world_size) zero matrix with the diagonal for the rank
    of this process set to 1.

    Example output where n=3, the current process has rank 1, and there are
    4 processes in total:

        rank0   rank1   rank2   rank3
        0 0 0 | 1 0 0 | 0 0 0 | 0 0 0
        0 0 0 | 0 1 0 | 0 0 0 | 0 0 0
        0 0 0 | 0 0 1 | 0 0 0 | 0 0 0

    Equivalent to torch.eye for undistributed settings or if world size == 1.

    Args:
        n:
            Size of the square matrix on a single process.
        device:
            Device on which the matrix should be created.

    Returns:
        A tensor with the appropriate diagonal filled for this rank.

    """
    rows = torch.arange(n, device=device, dtype=torch.long)
    cols = rows + rank() * n
    diag_mask = torch.zeros((n, n * world_size()), dtype=torch.bool)
    diag_mask[(rows, cols)] = True
    return diag_mask


R = TypeVar("R")


def rank_zero_only(fn: Callable[..., R]) -> Callable[..., Optional[R]]:
    """Decorator to ensure the function only runs on the process with rank 0.

    Example:
        >>> @rank_zero_only
        >>> def print_rank_zero(message: str):
        >>>     print(message)
        >>>
        >>> print_rank_zero("Hello from rank 0!")
    """

    def wrapped(*args: Any, **kwargs: Any) -> Optional[R]:
        if rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped


@rank_zero_only
def print_rank_zero(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
    """Equivalent to print, but only runs on the process with rank 0."""
    print(*args, **kwargs)
