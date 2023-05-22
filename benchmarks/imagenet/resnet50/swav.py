import math
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity, ModuleList
from torch.nn import functional as F
from torchvision.models import resnet50

from lightly.loss.memory_bank import MemoryBankModule
from lightly.loss.swav_loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.utils import get_weight_decay_parameters
from lightly.transforms import SwaVTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler

CROP_COUNTS: Tuple[int, int] = (2, 6)


class SwAV(LightningModule):
    def __init__(self, batch_size: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = SwaVProjectionHead()
        self.prototypes = SwaVPrototypes(
            n_steps_frozen_prototypes=(
                2# TODO: self.trainer.estimated_stepping_batches / self.trainer.max_epochs
            )
        )
        self.criterion = SwaVLoss(sinkhorn_gather_distributed=True)
        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

        if self.batch_size <= 256:
            self.start_queue_at_epoch = 15
            self.queues = ModuleList(
                [
                    MemoryBankModule(size=15 * self.batch_size)
                    for _ in range(CROP_COUNTS[0])
                ]
            )
        else:
            self.start_queue_at_epoch = None
            self.queues = None

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        # Normalize the prototypes so they are on the unit sphere.
        self.prototypes.normalize()

        # The multi-crop dataloader returns a list of image crops where the
        # first few items are high resolution crops and the rest are low
        # resolution crops.
        multi_crops, targets, _ = batch
        multi_crop_features = [
            self.forward(crops).flatten(start_dim=1) for crops in multi_crops
        ]

        def _project(x: Tensor) -> Tensor:
            x = self.projection_head(x)
            return F.normalize(x, dim=1, p=2)

        multi_crop_projections = [
            _project(features) for features in multi_crop_features
        ]
        multi_crop_logits = [
            self.prototypes(projections, step=self.global_step)
            for projections in multi_crop_projections
        ]

        # Get the queue projections and logits for small batch sizes (<= 256).
        queue_crop_logits = None
        if self.queues is not None and self.start_queue_at_epoch is not None:
            queue_crop_projections = _enqueue_and_get_queue_projections(
                high_resolution_projections=multi_crop_projections[: CROP_COUNTS[0]],
                queues=self.queues,
            )
            if self.current_epoch >= self.start_queue_at_epoch:
                with torch.no_grad():
                    queue_crop_logits = [
                        self.prototypes(projections, step=self.global_step)
                        for projections in queue_crop_projections
                    ]

        # Split the list of crop logits into high and low resolution.
        high_resolution_logits = multi_crop_logits[: CROP_COUNTS[0]]
        low_resolution_logits = multi_crop_logits[CROP_COUNTS[0] :]

        # Calculate the SwAV loss.
        loss = self.criterion(
            high_resolution_outputs=high_resolution_logits,
            low_resolution_outputs=low_resolution_logits,
            queue_outputs=queue_crop_logits,
        )
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        # Calculate the classification loss.
        features = torch.cat(multi_crop_features[: CROP_COUNTS[0]])
        cls_loss, cls_log = self.online_classifier.training_step(
            (features.detach(), targets.repeat(CROP_COUNTS[0])), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head, self.prototypes]
        )
        # After warmup, we use the cosine learning rate decay [40 , 44] witha final value of 0.0048
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
            # scaled by linearly by batch size to lr=4.8 for batch_size=2048.
            # See Appendix A.1. and A.6. in SwAV paper https://arxiv.org/pdf/2006.09882.pdf
            lr=0.6 * (self.batch_size * self.trainer.world_size) / 256,
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
                end_value=0.0006 * (self.batch_size * self.trainer.world_size) / 256,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


transform = SwaVTransform(crop_counts=CROP_COUNTS)


@torch.no_grad()
def _enqueue_and_get_queue_projections(
    high_resolution_projections: List[Tensor],
    queues: ModuleList,
):
    """Adds the high resolution projections to the queues and returns the queues."""

    if len(high_resolution_projections) != len(queues):
        raise ValueError(
            f"The number of queues ({len(queues)}) should be equal to the number of high "
            f"resolution inputs ({len(high_resolution_projections)})."
        )

    # Get the queue projections
    queue_projections = []
    for i in range(len(queues)):
        _, projections = queues[i](high_resolution_projections[i], update=True)
        # Queue projections are in (num_ftrs X queue_length) shape, while the high res
        # projections are in (batch_size X num_ftrs). Swap the axes for interoperability.
        projections = torch.permute(projections, (1, 0))
        queue_projections.append(projections)

    return queue_projections
