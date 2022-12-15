from torch import nn

from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes


class SwaV(nn.Module):
    def __init__(self, backbone, num_ftrs, out_dim, n_prototypes):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(num_ftrs, num_ftrs, out_dim)
        self.prototypes = SwaVPrototypes(out_dim, n_prototypes=n_prototypes)

    def forward(self, high_resolution, low_resolution):
        self.prototypes.normalize()
        high_resolution = [self._subforward(x) for x in high_resolution]
        low_resolution = [self._subforward(x) for x in low_resolution]
        return high_resolution, low_resolution

    def _subforward(self, input):
        features = self.backbone(input).flatten(start_dim=1)
        features = self.projection_head(features)
        features = nn.functional.normalize(features, dim=1, p=2)
        return self.prototypes(features)
