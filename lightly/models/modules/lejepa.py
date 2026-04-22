"""Reusable LeJEPA components: invariance loss, combined loss, and encoder."""

import torch
from torch import Tensor, nn

from lightly.loss.lejepa_loss import SIGReg
from lightly.models.modules.heads import LeJEPAProjectionHead


def lejepa_invariance_loss(proj: torch.Tensor) -> torch.Tensor:
    """LeJEPA invariance loss across multiple views.

    Pulls each view's projection toward the per-sample mean across views.
    Given projections of shape ``(V, N, D)``, this is the mean-squared
    distance between every view and the per-sample centroid computed over
    the view dimension.

    Reference:
        LeJEPA, 2025, https://arxiv.org/abs/2511.08544

    Args:
        proj:
            Projected embeddings of shape ``(V, N, D)`` where ``V`` is the
            number of views, ``N`` is the batch size, and ``D`` is the
            projection dimensionality.

    Returns:
        Scalar invariance loss.
    """
    return (proj.mean(0) - proj).square().mean()


class LeJEPALoss(nn.Module):
    """LeJEPA loss combining SIGReg regularization with view invariance.

    The loss is a convex combination of two terms:

    - ``SIGReg(proj)`` regularizes projections toward an isotropic Gaussian
      distribution.
    - ``lejepa_invariance_loss(proj)`` pulls each view toward the per-sample
      mean across views.

    The total loss is
    ``lambda_param * SIGReg(proj) + (1 - lambda_param) * invariance(proj)``.

    The default ``lambda_param=0.02`` matches the reference implementation
    [1]. The paper [0] explores values between 0.01 and 0.1.

    - [0]: LeJEPA, 2025, https://arxiv.org/abs/2511.08544
    - [1]: https://github.com/galilai-group/lejepa

    Attributes:
        lambda_param:
            Weight for the SIGReg term in [0, 1]. The invariance term is
            weighted by (1 - lambda_param).
        sigreg:
            The SIGReg regularization module instantiated with the supplied
            sigreg_* parameters.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = LeJEPALoss()
        >>>
        >>> # generate multiple views of the same images
        >>> views = [transform(images) for _ in range(n_views)]
        >>>
        >>> # project each view and stack to shape (V, N, D)
        >>> proj = torch.stack([model(v) for v in views])
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(proj)
    """

    def __init__(
        self,
        lambda_param: float = 0.02,
        gather_distributed: bool = False,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_num_vectors: int = 256,
    ):
        """Initialize the combined LeJEPA loss.

        Args:
            lambda_param:
                Weight for the SIGReg term in ``[0, 1]``. The invariance
                term is weighted by ``1 - lambda_param``.
            gather_distributed:
                If True, aggregate SIGReg statistics across distributed
                ranks so the loss uses the global batch.
            sigreg_knots:
                Number of frequency samples used for the SIGReg integration
                grid.
            sigreg_t_max:
                Maximum frequency for the SIGReg integration grid.
            sigreg_num_vectors:
                Number of random unit projection vectors (slices) used by
                SIGReg per forward pass.
        """
        super().__init__()
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must be in the unit interval [0, 1].")

        self.lambda_param = lambda_param
        self.sigreg = SIGReg(
            knots=sigreg_knots,
            t_max=sigreg_t_max,
            num_vectors=sigreg_num_vectors,
            gather_distributed=gather_distributed,
        )

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute the LeJEPA loss for a batch of multi-view projections.

        Args:
            proj: Projected embeddings of shape ``(V, N, D)``.
        """
        sigreg_loss = self.sigreg(proj)
        inv_loss = lejepa_invariance_loss(proj)
        loss: torch.Tensor = (
            self.lambda_param * sigreg_loss + (1.0 - self.lambda_param) * inv_loss
        )
        return loss


class LeJEPAEncoder(nn.Module):
    """Encoder wrapping a backbone with a LeJEPA projection head.

    Convenience module that bundles a feature-extraction backbone with a
    ``LeJEPAProjectionHead`` so a single ``forward`` call maps input
    images to projected embeddings. Backbone outputs are flattened from
    the second dimension onward before being fed to the projection head,
    which matches the pattern used in the other LeJEPA examples.

    - [0]: LeJEPA, 2025, https://arxiv.org/abs/2511.08544

    Attributes:
        backbone:
            Feature-extraction module. The output is flattened from
            ``dim=1`` onward before being passed to the projection head.
        projection_head:
            ``LeJEPAProjectionHead`` producing the final projected
            embeddings.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projection_head: LeJEPAProjectionHead,
    ) -> None:
        """Initializes the LeJEPAEncoder with a backbone and projection head.

        Args:
            backbone:
                Feature-extraction module.
            projection_head:
                ``LeJEPAProjectionHead`` applied to the flattened backbone
                outputs.
        """
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head

    def forward(self, x: Tensor) -> Tensor:
        """Projects the input through backbone and projection head.

        Args:
            x:
                Input tensor of shape ``(N, ...)`` that the backbone
                accepts.

        Returns:
            Projected embeddings of shape ``(N, output_dim)``.
        """
        features = self.backbone(x).flatten(start_dim=1)
        projections: Tensor = self.projection_head(features)
        return projections
