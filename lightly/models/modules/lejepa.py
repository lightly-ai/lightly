"""LeJEPA encoder module."""

from torch import Tensor, nn

from lightly.models.modules.heads import LeJEPAProjectionHead


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
