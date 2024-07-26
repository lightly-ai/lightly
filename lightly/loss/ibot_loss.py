import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class iBOTPatchLoss(Module):
    """Implementation of the iBOT patch loss [0] as used in DINOv2 [1].

    Implementation is based on [2].

    - [0]: iBOT, 2021, https://arxiv.org/abs/2111.07832
    - [1]: DINOv2, 2023, https://arxiv.org/abs/2304.07193
    - [2]: https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py

    Attributes:
        temperature:
            Temperature of the softmax for the student outputs.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        self.temperature = temperature

    def forward(self, teacher_out: Tensor, student_out: Tensor, mask: Tensor) -> Tensor:
        """Forward pass through the iBOT patch loss.

        Args:
            teacher_out:
                Tensor with shape (B * N, D) containing the teacher output of the masked
                tokens.
            student_out:
                Tensor with shape (B * N, D) containing the student output of the masked
                tokens.
            mask:
                Boolean tensor with shape (B, H, W) containing the token mask.
                Exactly B * N entries must be set to True in the mask.
        Returns:
            Loss value.
        """
        # TODO: Add centering.

        # Calculate cross entropy loss.
        student_log_softmax = F.log_softmax(student_out / self.temperature, dim=-1)
        # (B * N, D) -> (B * N)
        loss = -torch.sum(teacher_out * student_log_softmax, dim=-1)

        # Get weights.
        # (B, H, W) -> (B, 1, 1)
        num_masked_per_image = (
            mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0).clamp(min=1.0)
        )
        # (B, 1, 1) -> (B, H, W) -> (B * N)
        weight = (1.0 / num_masked_per_image).expand_as(mask)[mask]

        # Apply weighthing.
        B = mask.shape[0]
        return (loss * weight).sum() / B
