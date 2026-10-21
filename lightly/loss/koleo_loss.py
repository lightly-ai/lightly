from __future__ import annotations

import torch
import torch.distributed as torch_dist
from torch import Tensor
from torch.nn import Module, PairwiseDistance, functional

from lightly.utils import dist as lightly_dist


@torch.no_grad()
def _nearest_neighbor_indices(x: Tensor, group_size: int, topk: int) -> Tensor:
    """Returns the indices of the topk nearest neighbors within every group.

    The batch is split into consecutive groups of group_size features and neighbors
    are searched by cosine similarity within each group. A feature is never its own
    neighbor unless the group holds no other feature.

    Args:
        x:
            Tensor with shape (batch_size, embedding_size) containing L2-normalized
            features. The batch size must be divisible by group_size.
        group_size:
            Number of features per group.
        topk:
            Number of neighbors per feature.

    Returns:
        Tensor with shape (batch_size * topk,) containing indices into x, ordered by
        feature and then by neighbor.
    """
    # (batch_size, embedding_size) -> (num_groups, group_size, embedding_size)
    x_grouped = x.view(-1, group_size, x.shape[1])
    num_groups = x_grouped.shape[0]

    # Cosine similarity within every group, with self-similarity masked out.
    cos_sim = torch.bmm(x_grouped, x_grouped.transpose(1, 2))
    cos_sim.diagonal(dim1=-2, dim2=-1).fill_(-2)

    # (num_groups, group_size, topk)
    nn_idx = cos_sim.topk(k=topk, dim=-1).indices

    # Shift the group-local indices so that they index into the flat batch.
    offset = torch.arange(num_groups, device=x.device) * group_size
    return (nn_idx + offset.view(-1, 1, 1)).flatten()


class KoLeoLoss(Module):
    """KoLeo loss based on [0].

    KoLeo loss is a regularizer that encourages a uniform span of the features in a
    batch by penalizing the distance between the features and their nearest
    neighbors.

    Implementation is based on [1]. Nearest neighbors are searched within groups of
    group_size features, following the distributed KoLeo loss used by DINOv3 [2][3].

    - [0]: Spreading vectors for similarity search, 2019, https://arxiv.org/abs/1806.03198
    - [1]: https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
    - [2]: DINOv3, 2025, https://arxiv.org/abs/2508.10104
    - [3]: https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/koleo_loss.py

    Attributes:
        p:
            The norm degree for pairwise distance calculation.
        eps:
            Small value to avoid division by zero.
        topk:
            Number of nearest neighbors per feature that are penalized.
        group_size:
            Number of features within which nearest neighbors are searched. The batch
            is split into consecutive groups of this size. If None, the whole batch is
            used as a single group. DINOv3 uses a group size of 16.
        gather_distributed:
            If True, features from all GPUs are gathered before the batch is split into
            groups. Has no effect if the distributed process group is not initialized.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = KoLeoLoss()
        >>>
        >>> # or with the settings used by DINOv3
        >>> loss_fn = KoLeoLoss(group_size=16, gather_distributed=True)
        >>>
        >>> # generate the features of a batch of images
        >>> features = model(images)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(features)
    """

    def __init__(
        self,
        p: float = 2,
        eps: float = 1e-8,
        topk: int = 1,
        group_size: int | None = None,
        gather_distributed: bool = False,
    ):
        """Initializes the KoLeoLoss module with the specified parameters.

        Args:
            p:
                The norm degree for pairwise distance calculation.
            eps:
                Small value to avoid division by zero.
            topk:
                Number of nearest neighbors per feature that are penalized.
            group_size:
                Number of features within which nearest neighbors are searched. The
                batch is split into consecutive groups of this size. If None, the whole
                batch is used as a single group. DINOv3 uses a group size of 16.
            gather_distributed:
                If True, features from all GPUs are gathered before the batch is split
                into groups. Has no effect if the distributed process group is not
                initialized.

        Raises:
            ValueError: If topk or group_size are not positive.
            ValueError: If gather_distributed is True but torch.distributed is not
                available.
        """
        super().__init__()
        if topk < 1:
            raise ValueError(f"topk must be positive but is {topk}.")
        if group_size is not None and group_size < 1:
            raise ValueError(f"group_size must be positive but is {group_size}.")
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

        self.p = p
        self.eps = eps
        self.topk = topk
        self.group_size = group_size
        self.gather_distributed = gather_distributed
        self.pairwise_distance = PairwiseDistance(p=p, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through KoLeo Loss.

        Args:
            x: Tensor with shape (batch_size, embedding_size).

        Returns:
            Loss value.

        Raises:
            ValueError: If the batch is empty, if the batch size is not divisible by
                group_size, or if topk is too large for the group size.
        """
        # Normalize the input tensor
        x = functional.normalize(x, p=2, dim=-1, eps=self.eps)

        # Gather features from all GPUs. The loss is calculated over the global batch
        # on every process, gradients are averaged by the GatherLayer.
        if self.gather_distributed and lightly_dist.world_size() > 1:
            x = torch.cat(lightly_dist.gather(x), dim=0)

        batch_size = x.shape[0]
        if batch_size == 0:
            raise ValueError("KoLeoLoss requires a non-empty batch.")

        group_size = self.group_size if self.group_size is not None else batch_size
        if batch_size % group_size != 0:
            raise ValueError(
                f"Batch size {batch_size} must be divisible by group size {group_size}."
            )
        # A group must hold topk neighbors besides the feature itself. Groups of size
        # one are the exception: there the feature is its own neighbor, which keeps
        # a batch size of one working as it did before groups existed.
        max_topk = max(group_size - 1, 1)
        if self.topk > max_topk:
            raise ValueError(
                f"topk {self.topk} must not be larger than {max_topk} for group size "
                f"{group_size}."
            )

        # Get the nearest neighbors and their distances.
        nn_idx = _nearest_neighbor_indices(x=x, group_size=group_size, topk=self.topk)
        nn_dist: Tensor = self.pairwise_distance(
            x.repeat_interleave(self.topk, dim=0), x[nn_idx]
        )

        # Compute the loss
        loss = -(nn_dist + self.eps).log().mean()

        return loss
