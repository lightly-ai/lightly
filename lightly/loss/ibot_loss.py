import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from lightly.models.modules.center import Center


class IBOTPatchLoss(Module):
    """Implementation of the iBOT patch loss [0] as used in DINOv2 [1].

    Implementation is based on [2].

    - [0]: iBOT, 2021, https://arxiv.org/abs/2111.07832
    - [1]: DINOv2, 2023, https://arxiv.org/abs/2304.07193
    - [2]: https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py

    Attributes:
        output_dim:
            Dimension of the model output.
        teacher_temp:
            Temperature for the teacher output.
        student_temp:
            Temperature for the student output.
        center_mode:
            Mode for center calculation. Only 'mean' is supported.
        center_momentum:
            Momentum term for the center update.
    """

    def __init__(
        self,
        output_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_mode: str = "mean",
        center_momentum: float = 0.9,
    ) -> None:
        """Initializes the iBOTPatchLoss module with the specified parameters.

        Args:
            output_dim:
                Dimension of the model output.
            teacher_temp:
                Temperature for the teacher output.
            student_temp:
                Temperature for the student output.
            center_mode:
                Mode for center calculation. Only 'mean' is supported.
            center_momentum:
                Momentum term for the center update.
        """
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temperature = student_temp
        self.center = Center(
            size=(1, output_dim),
            mode=center_mode,
            momentum=center_momentum,
        )

    def forward(
        self,
        teacher_out: Tensor,
        student_out: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Forward pass through the iBOT patch loss.

        Args:
            teacher_out:
                Tensor with shape (batch_size * sequence_length, embed_dim) containing
                the teacher output of the masked tokens.
            student_out:
                Tensor with shape (batch_size * sequence_length, embed_dim) containing
                the student output of the masked tokens.
            mask:
                Boolean tensor with shape (batch_size, height, width) containing the
                token mask. Exactly batch_size * sequence_length entries must be set to
                True in the mask.

        Returns:
            The loss value.
        """
        # B = batch size, N = sequence length = number of masked tokens, D = embed dim
        # H = height (in tokens), W = width (in tokens)
        # Note that N <= H * W depending on how many tokens are masked.

        # Calculate cross-entropy loss.
        teacher_softmax = F.softmax(
            (teacher_out - self.center.value) / self.teacher_temp, dim=-1
        )
        student_log_softmax = F.log_softmax(
            student_out / self.student_temperature, dim=-1
        )
        # (B * N, D) -> (B * N)
        loss = -torch.sum(teacher_softmax * student_log_softmax, dim=-1)

        # Get weights.
        # (B, H, W) -> (B, 1, 1)
        num_masked_per_image = mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        # (B, 1, 1) -> (B, H, W) -> (B * N)
        weight = (1.0 / num_masked_per_image).expand_as(mask)[mask]

        # Apply weighting.
        B = mask.shape[0]
        loss = (loss * weight).sum() / B

        self.center.update(teacher_out)
        return loss
