import torch
from torch import nn

from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.loss.memory_bank import MemoryBankModule


class SwaV(nn.Module):
    def __init__(self, backbone, num_ftrs, out_dim, n_prototypes, queue_length=0):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(num_ftrs, num_ftrs, out_dim)
        self.prototypes = SwaVPrototypes(out_dim, n_prototypes=n_prototypes)
        # Queues are initialized in the first forward call
        self.queues = []
        self.queue_length = queue_length

    def forward(self, high_resolution, low_resolution):
        self._create_queues(n_queues=len(high_resolution), device=high_resolution[0].device)

        self.prototypes.normalize()

        high_resolution_features = [self._subforward(x) for x in high_resolution]
        low_resolution_features = [self._subforward(x) for x in low_resolution]
        queue_features = self._get_queue_features(high_resolution_features)

        high_resolution_prototypes = [self.prototypes(x) for x in high_resolution_features]
        low_resolution_prototypes = [self.prototypes(x) for x in low_resolution_features]
        queue_prototypes = [self.prototypes(x) for x in queue_features]

        return high_resolution_prototypes, low_resolution_prototypes, queue_prototypes

    def _subforward(self, input):
        features = self.backbone(input).flatten(start_dim=1)
        features = self.projection_head(features)
        features = nn.functional.normalize(features, dim=1, p=2)
        return features

    def _create_queues(self, n_queues, device):
        # Create one queue for each high resolution view
        if not self.queues and self.queue_length > 0:
            for i in range(n_queues):
                queue = MemoryBankModule(size=self.queue_length)
                self.queues.append(queue.to(device))

    def _get_queue_features(self, high_resolution_features):
        queue_features = []
        if self.queue_length > 0:
            with torch.no_grad():
                for i in range(len(high_resolution_features)):
                    queue = self.queues[i]
                    features = high_resolution_features[i]
                    queue_features.append(queue(features, update=True)[1])
        return queue_features
    