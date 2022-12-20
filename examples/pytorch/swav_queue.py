# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
import torchvision
from torch import nn

from lightly.data import LightlyDataset, SwaVCollateFunction
from lightly.loss import SwaVLoss
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes


class SwaV(nn.Module):
    def __init__(self, backbone, num_ftrs, out_dim, n_prototypes,
                 n_queues, queue_length=0, start_queue_at_epoch=0):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(num_ftrs, num_ftrs, out_dim)
        self.prototypes = SwaVPrototypes(out_dim, n_prototypes=n_prototypes)

        self.start_queue_at_epoch = start_queue_at_epoch
        self.queues = None
        if n_queues > 0:
            self.queues = [MemoryBankModule(size=queue_length) for _ in range(n_queues)]
            self.queues = nn.ModuleList(self.queues)

    def forward(self, high_resolution, low_resolution, epoch=None):
        self.prototypes.normalize()

        high_resolution_features = [self._subforward(x) for x in high_resolution]
        low_resolution_features = [self._subforward(x) for x in low_resolution]

        high_resolution_prototypes = [self.prototypes(x) for x in high_resolution_features]
        low_resolution_prototypes = [self.prototypes(x) for x in low_resolution_features]
        queue_prototypes = self._get_queue_prototypes(high_resolution_features, epoch)

        return high_resolution_prototypes, low_resolution_prototypes, queue_prototypes

    def _subforward(self, input):
        features = self.backbone(input).flatten(start_dim=1)
        features = self.projection_head(features)
        features = nn.functional.normalize(features, dim=1, p=2)
        return features

    @torch.no_grad()
    def _get_queue_prototypes(self, high_resolution_features, epoch=None):
        if self.queues is None:
            return None

        if len(high_resolution_features) != len(self.queues):
            raise ValueError(
                f"The number of queues ({len(self.queues)}) should be equal to the number of high "
                f"resolution inputs ({len(high_resolution_features)}). Set `n_queues` accordingly."
            )

        # Get the queue features
        queue_features = []
        for i in range(len(self.queues)):
            _, features = self.queues[i](high_resolution_features[i], update=True)
            # Queue features are in (num_ftrs X queue_length) shape, while the high res
            # features are in (batch_size X num_ftrs). Swap the axes for interoperability.
            features = torch.permute(features, (1,0))
            queue_features.append(features)
        
        # If loss calculation with queue prototypes starts at a later epoch,
        # just queue the features and return None instead of queue prototypes.
        if self.start_queue_at_epoch > 0:
            if epoch is None:
                raise ValueError("The epoch number must be passed to the `forward()` "
                                 "method if `start_queue_at_epoch` is greater than 0.")
            if epoch < self.start_queue_at_epoch:
                return None

        # Assign prototypes
        queue_prototypes = [self.prototypes(x) for x in queue_features]
        return queue_prototypes


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SwaV(backbone, num_ftrs=512, out_dim=128, n_prototypes=512,
             n_queues=2, queue_length=512, start_queue_at_epoch=5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# we ignore object detection annotations by setting target_transform to return 0
pascal_voc = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc", download=True, target_transform=lambda t: 0
)
dataset = LightlyDataset.from_torch_dataset(pascal_voc)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = SwaVCollateFunction()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = SwaVLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for batch, _, _ in dataloader:
        batch = [x.to(device) for x in batch]
        high_resolution, low_resolution = batch[:2], batch[2:]
        high_resolution, low_resolution, queue = model(high_resolution, low_resolution, epoch)
        loss = criterion(high_resolution, low_resolution, queue)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
