# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
from torch import nn
import torchvision
import pytorch_lightning as pl

from lightly.data import LightlyDataset
from lightly.data import SwaVCollateFunction
from lightly.loss import SwaVLoss
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.modules import SwaVProjectionHead
from lightly.models.modules import SwaVPrototypes


class SwaV(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SwaVProjectionHead(512, 512, 128)
        self.prototypes = SwaVPrototypes(128, 512, 1)
        self.start_queue_at_epoch = 2
        self.queues = nn.ModuleList([MemoryBankModule(size=3840) for _ in range(2)])
        self.criterion = SwaVLoss()

    def training_step(self, batch, batch_idx):
        batch_swav, _, _ = batch
        high_resolution, low_resolution = batch_swav[:2], batch_swav[2:]
        self.prototypes.normalize()

        high_resolution_features = [self._subforward(x) for x in high_resolution]
        low_resolution_features = [self._subforward(x) for x in low_resolution]

        high_resolution_prototypes = [
            self.prototypes(x, self.current_epoch) for x in high_resolution_features
        ]
        low_resolution_prototypes = [
            self.prototypes(x, self.current_epoch) for x in low_resolution_features
        ]
        queue_prototypes = self._get_queue_prototypes(high_resolution_features)
        loss = self.criterion(
            high_resolution_prototypes, low_resolution_prototypes, queue_prototypes
        )
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim

    def _subforward(self, input):
        features = self.backbone(input).flatten(start_dim=1)
        features = self.projection_head(features)
        features = nn.functional.normalize(features, dim=1, p=2)
        return features

    @torch.no_grad()
    def _get_queue_prototypes(self, high_resolution_features):

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
            features = torch.permute(features, (1, 0))
            queue_features.append(features)

        # If loss calculation with queue prototypes starts at a later epoch,
        # just queue the features and return None instead of queue prototypes.
        if (
            self.start_queue_at_epoch > 0
            and self.current_epoch < self.start_queue_at_epoch
        ):
            return None

        # Assign prototypes
        queue_prototypes = [self.prototypes(x, self.current_epoch) for x in queue_features]
        return queue_prototypes


model = SwaV()

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

gpus = 1 if torch.cuda.is_available() else 0

trainer = pl.Trainer(max_epochs=10, gpus=gpus)
trainer.fit(model=model, train_dataloaders=dataloader)
