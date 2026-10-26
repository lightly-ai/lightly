"""CAPI loss.

- [0]: CAPI: Cluster and Predict Latent Patches for Improved Masked Image Modeling, 2025, https://arxiv.org/abs/2502.08769
- [1]: https://github.com/facebookresearch/capi
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


@torch.no_grad()
def sinkhorn_knopp(
    logits: Tensor,
    iterations: int = 3,
    gather_distributed: bool = False,
    epsilon: float = 1e-8,
) -> Tensor:
    """Positionwise Sinkhorn-Knopp normalization of cluster logits as used in CAPI [0].

    The logits are normalized independently for each sequence position: within a
    position the assignment mass is balanced uniformly over the batch, while each
    token keeps a distribution over the clusters. Balancing per position rather
    than globally avoids the positional collapse described in [0].

    - [0]: CAPI: Cluster and Predict Latent Patches for Improved Masked Image Modeling, 2025, https://arxiv.org/abs/2502.08769

    Args:
        logits:
            Cluster logits with shape (batch_size, sequence_length, num_clusters).
        iterations:
            Number of Sinkhorn-Knopp iterations.
        gather_distributed:
            If True, the batch normalization is synchronized across all processes.
        epsilon:
            Small value added to the denominators for numerical stability.

    Returns:
        Soft cluster assignments with the same shape as the input; each token's
        assignment is a distribution over the clusters.
    """
    world_size = 1
    if gather_distributed and dist.is_initialized():
        world_size = dist.get_world_size()

    # Move the sequence axis to the front so each position is normalized on its own,
    # balancing over the batch (dim=1) and the clusters (dim=2).
    q = logits.transpose(0, 1)  # (sequence_length, batch_size, num_clusters)

    # Numerically stable exponential: shift each column by its max before exp. The
    # shift cancels in the following normalization, so it does not affect the result.
    shift = torch.amax(q, dim=1, keepdim=True)
    if world_size > 1:
        dist.all_reduce(shift, op=dist.ReduceOp.MAX)
    q = torch.exp(q - shift)

    for _ in range(iterations):
        sum_over_batch = torch.sum(q, dim=1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(sum_over_batch)
        q = q / (sum_over_batch + epsilon)
        q = q / (torch.sum(q, dim=2, keepdim=True) + epsilon)

    return q.transpose(0, 1)  # (batch_size, sequence_length, num_clusters)


class CAPILoss(Module):
    """Implementation of the CAPI loss [0].

    The teacher's cluster logits are turned into soft targets with a positionwise
    Sinkhorn-Knopp normalization, and the student is trained to predict these
    targets with a cross-entropy loss. The same objective trains the online
    clustering prototypes when the teacher logits are passed as ``student_logits``,
    as done in the reference implementation [1].

    - [0]: CAPI: Cluster and Predict Latent Patches for Improved Masked Image Modeling, 2025, https://arxiv.org/abs/2502.08769
    - [1]: https://github.com/facebookresearch/capi

    Attributes:
        teacher_temperature:
            Temperature applied to the teacher logits before the Sinkhorn-Knopp
            normalization.
        student_temperature:
            Temperature applied to the student logits in the cross-entropy.
        sinkhorn_iterations:
            Number of Sinkhorn-Knopp iterations.
        gather_distributed:
            If True, the Sinkhorn normalization is synchronized across all processes.

    Raises:
        ValueError: If gather_distributed is True but torch.distributed is not
            available.
    """

    def __init__(
        self,
        teacher_temperature: float = 0.06,
        student_temperature: float = 0.12,
        sinkhorn_iterations: int = 3,
        gather_distributed: bool = False,
    ) -> None:
        super().__init__()
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or use a distributed-enabled "
                "installation of PyTorch."
            )
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.gather_distributed = gather_distributed

    def forward(
        self,
        teacher_logits: Tensor,
        student_logits: Tensor,
        teacher_index: Tensor | None = None,
    ) -> Tensor:
        """Computes the CAPI cross-entropy loss over the predicted tokens.

        The Sinkhorn-Knopp normalization is applied over all tokens of
        ``teacher_logits``. When ``teacher_index`` is given, the teacher targets
        are selected at those positions after the normalization, so the balancing
        spans the full set of teacher tokens rather than only the predicted subset.

        Args:
            teacher_logits:
                Teacher cluster logits with shape
                (batch_size, num_teacher_tokens, num_clusters).
            student_logits:
                Student cluster logits of the predicted tokens with shape
                (batch_size, num_predicted_tokens, num_clusters).
            teacher_index:
                Optional row-major indices of the predicted tokens into the
                teacher tokens with shape (batch_size, num_predicted_tokens). If
                given, the teacher targets are gathered at these positions after
                the Sinkhorn-Knopp normalization; otherwise ``teacher_logits`` and
                ``student_logits`` are assumed to already be aligned.

        Returns:
            The mean cross-entropy loss over all predicted tokens.
        """
        # Teacher targets are detached, so no gradient flows through them.
        assignments = sinkhorn_knopp(
            logits=teacher_logits.detach() / self.teacher_temperature,
            iterations=self.sinkhorn_iterations,
            gather_distributed=self.gather_distributed,
        )
        if teacher_index is not None:
            # Select the teacher targets at the predicted positions.
            index = teacher_index.unsqueeze(-1).expand(-1, -1, assignments.shape[-1])
            assignments = torch.gather(assignments, 1, index)
        log_probs = F.log_softmax(student_logits / self.student_temperature, dim=-1)
        loss = -torch.sum(assignments * log_probs, dim=-1)
        return loss.mean()
