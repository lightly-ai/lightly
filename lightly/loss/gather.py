from typing import Tuple

import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.
    
    This code was taken and adapted from here:
    https://github.com/Spijkervet/SimCLR
    
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.save_for_backward(input)
        output = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        grad_out = torch.empty_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out