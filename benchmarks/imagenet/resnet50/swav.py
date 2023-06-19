import math
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity, ModuleList
from torch.nn import functional as F
from torchvision.models import resnet50

from lightly.models.modules.memory_bank import MemoryBankModule
from lightly.loss.swav_loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.utils import get_weight_decay_parameters
from lightly.transforms import SwaVTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler

CROP_COUNTS: Tuple[int, int] = (2, 6)


class SwAV(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = SwaVProjectionHead()
        self.prototypes = SwaVPrototypes(n_steps_frozen_prototypes=1)
        self.criterion = SwaVLoss(sinkhorn_gather_distributed=True)
        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

        # Use a queue for small batch sizes (<= 256).
        self.start_queue_at_epoch = 15
        self.n_batches_in_queue = 15
        self.queues = ModuleList(
            [
                MemoryBankModule(
                    size=self.n_batches_in_queue * self.batch_size_per_device
                )
                for _ in range(CROP_COUNTS[0])
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def project(self, x: Tensor) -> Tensor:
        x = self.projection_head(x)
        return F.normalize(x, dim=1, p=2)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        # Normalize the prototypes so they are on the unit sphere.
        self.prototypes.normalize()

        # The dataloader returns a list of image crops where the
        # first few items are high resolution crops and the rest are low
        # resolution crops.
        multi_crops, targets = batch[0], batch[1]

        # Forward pass through backbone and projection head.
        multi_crop_features = [
            self.forward(crops).flatten(start_dim=1) for crops in multi_crops
        ]
        multi_crop_projections = [
            self.project(features) for features in multi_crop_features
        ]

        # Get the queue projections and logits.
        queue_crop_logits = None
        with torch.no_grad():
            if self.current_epoch >= self.start_queue_at_epoch:
                # Start filling the queue.
                queue_crop_projections = _update_queue(
                    projections=multi_crop_projections[: CROP_COUNTS[0]],
                    queues=self.queues,
                )
                if batch_idx > self.n_batches_in_queue:
                    # The queue is filled, so we can start using it.
                    queue_crop_logits = [
                        self.prototypes(projections, step=self.current_epoch)
                        for projections in queue_crop_projections
                    ]

        # Get the rest of the multi-crop logits.
        multi_crop_logits = [
            self.prototypes(projections, step=self.current_epoch)
            for projections in multi_crop_projections
        ]

        # Calculate the SwAV loss.
        loss = self.criterion(
            high_resolution_outputs=multi_crop_logits[: CROP_COUNTS[0]],
            low_resolution_outputs=multi_crop_logits[CROP_COUNTS[0] :],
            queue_outputs=queue_crop_logits,
        )
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size_per_device=len(targets),
        )

        # Calculate the classification loss.
        cls_loss, cls_log = self.online_classifier.training_step(
            (multi_crop_features[0].detach(), targets), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size_per_device=len(targets))
        return loss + cls_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(
            cls_log, prog_bar=True, sync_dist=True, batch_size_per_device=len(targets)
        )
        return cls_loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head, self.prototypes]
        )
        optimizer = LARS(
            [
                {"name": "swav", "params": params},
                {
                    "name": "swav_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Smaller learning rate for smaller batches: lr=0.6 for batch_size=256
            # scaled linearly by batch size to lr=4.8 for batch_size=2048.
            # See Appendix A.1. and A.6. in SwAV paper https://arxiv.org/pdf/2006.09882.pdf
            lr=0.6 * (self.batch_size_per_device * self.trainer.world_size) / 256,
            momentum=0.9,
            weight_decay=1e-6,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=self.trainer.estimated_stepping_batches,
                end_value=0.0006
                * (self.batch_size_per_device * self.trainer.world_size)
                / 256,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


transform = SwaVTransform(crop_counts=CROP_COUNTS)


@torch.no_grad()
def _update_queue(
    projections: List[Tensor],
    queues: ModuleList,
):
    """Adds the high resolution projections to the queues and returns the queues."""

    if len(projections) != len(queues):
        raise ValueError(
            f"The number of queues ({len(queues)}) should be equal to the number of high "
            f"resolution inputs ({len(projections)})."
        )

    # Get the queue projections
    queue_projections = []
    for i in range(len(queues)):
        _, projections = queues[i](projections[i], update=True)
        # Queue projections are in (num_ftrs X queue_length) shape, while the high res
        # projections are in (batch_size_per_device X num_ftrs). Swap the axes for interoperability.
        projections = torch.permute(projections, (1, 0))
        queue_projections.append(projections)

    return queue_projections
