import torch
from torch import Tensor
from torch.nn import Module, PairwiseDistance, functional


class KoLeoLoss(Module):
    """KoLeo loss based on [0].

    KoLeo loss is a regularizer that encourages a uniform span of the features in a
    batch by penalizing the distance between the features and their nearest
    neighbors.

    Implementation is based on [1].

    - [0]: Spreading vectors for similarity search, 2019, https://arxiv.org/abs/1806.03198
    - [1]: https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py

    Attributes:
        p:
            The norm degree for pairwise distance calculation.
        eps:
            Small value to avoid division by zero.
    """

    def __init__(
        self,
        p: float = 2,
        eps: float = 1e-8,
    ):
        """Initializes the KoLeoLoss module with the specified parameters.

        Args:
            p:
                The norm degree for pairwise distance calculation.
            eps:
                Small value to avoid division by zero.
        """

        super().__init__()
        self.p = p
        self.eps = eps
        self.pairwise_distance = PairwiseDistance(p=p, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through KoLeo Loss.

        Args:
            x: Tensor with shape (batch_size, embedding_size).

        Returns:
            Loss value.
        """
        # Normalize the input tensor
        x = functional.normalize(x, p=2, dim=-1, eps=self.eps)

        # Calculate cosine similarity.
        cos_sim = torch.mm(x, x.t())
        cos_sim.fill_diagonal_(-2)

        # Get nearest neighbors.
        nn_idx = cos_sim.argmax(dim=1)
        nn_dist: Tensor = self.pairwise_distance(x, x[nn_idx])

        # Compute the loss
        loss = -(nn_dist + self.eps).log().mean()

        return loss
