from typing import Any, Callable, Optional, Tuple, TypeVar, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.autograd import Function


class GatherLayer(Function):
    """Gather tensors from all processes, supporting backward propagation.

    This code was taken and adapted from here:
    https://github.com/Spijkervet/SimCLR

    """

    # Type ignore misc is required because the superclass uses Any type for ctx.
    # Type ignore override is required because the superclass has a different signature
    # for forward.
    @staticmethod
    def forward(ctx: Any, input: Tensor) -> Tuple[Tensor, ...]:  # type: ignore[misc, override]
        ctx.save_for_backward(input)
        output = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    # Type ignore is required because superclass uses Any type for ctx.
    @staticmethod
    def backward(ctx: Any, *grads: Tensor) -> Tensor:  # type: ignore[misc]
        (input,) = ctx.saved_tensors
        grad_out = torch.empty_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def rank() -> int:
    """Returns the rank of the current process."""
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    """Returns the current world size (number of distributed processes)."""
    return dist.get_world_size() if dist.is_initialized() else 1


def gather(input: Tensor) -> Tuple[Tensor]:
    """Gathers this tensor from all processes. Supports backprop."""
    # Type ignore is required because Function.apply is untyped.
    return GatherLayer.apply(input)  # type: ignore


def eye_rank(n: int, device: Optional[torch.device] = None) -> Tensor:
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

    """
    rows = torch.arange(n, device=device, dtype=torch.long)
    cols = rows + rank() * n
    diag_mask = torch.zeros((n, n * world_size()), dtype=torch.bool)
    diag_mask[(rows, cols)] = True
    return diag_mask


_T = TypeVar("_T")


# TODO(Guarin, 01/2024): Refine typings for callable with ParamSpec once we drop support
# for Python <=3.9.
def rank_zero_only(fn: Callable[..., _T]) -> Callable[..., Union[_T, None]]:
    """Decorator that only runs the function on the process with rank 0.

    Example:
        >>> @rank_zero_only
        >>> def print_rank_zero(message: str):
        >>>     print(message)
        >>>
        >>> print_rank_zero("Hello from rank 0!")

    """

    def wrapped(*args: Any, **kwargs: Any) -> Union[_T, None]:
        if rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped


# Type ignore required because 'file' has Any type.
@rank_zero_only
def print_rank_zero(  # type: ignore[misc]
    *values: object,
    sep: Union[str, None] = " ",
    end: Union[str, None] = "\n",
    file: Any = None,
    flush: bool = False
) -> None:
    """Equivalent to print, but only runs on the process with rank 0."""
    print(*values, sep=sep, end=end, file=file, flush=flush)
