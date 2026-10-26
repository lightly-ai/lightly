from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from timm.models.vision_transformer import vit_small_patch16_224
from torch import Tensor
from torch.optim import AdamW

from lightly.loss import LeJEPALoss
from lightly.models.modules import LeJEPAProjectionHead
from lightly.transforms import DINOTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler


class LeJEPA(LightningModule):
    def __init__(
        self,
        batch_size_per_device: int,
        num_classes: int,
        lr: float = 5e-4,
        weight_decay: float = 5e-2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device
        self.lr = lr
        self.weight_decay = weight_decay

        self.backbone = vit_small_patch16_224(
            pretrained=False,
            pos_embed="learn",
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=0.1,
        )
        self.projection_head = LeJEPAProjectionHead(input_dim=self.backbone.embed_dim)
        self.criterion = LeJEPALoss(gather_distributed=True)
        self.online_classifier = OnlineLinearClassifier(
            feature_dim=self.backbone.embed_dim, num_classes=num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        views, targets = batch[0], batch[1]
        global_views = views[:2]
        local_views = views[2:]

        global_features = self.forward(torch.cat(global_views, dim=0))
        local_features = self.forward(torch.cat(local_views, dim=0))

        global_proj = torch.stack(
            self.projection_head(global_features).chunk(len(global_views))
        )
        local_proj = torch.stack(
            self.projection_head(local_features).chunk(len(local_views))
        )

        loss = self.criterion(local_proj=local_proj, global_proj=global_proj)
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        cls_loss, cls_log = self.online_classifier.training_step(
            (global_features.chunk(len(global_views))[0].detach(), targets), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        cls_features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (cls_features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        optimizer = AdamW(
            [
                {
                    "name": "lejepa",
                    "params": list(self.backbone.parameters())
                    + list(self.projection_head.parameters()),
                    "weight_decay": self.weight_decay,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=self.lr,
        )

        warmup_epochs = max(
            1, int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs)
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=int(self.trainer.estimated_stepping_batches),
                warmup_start_value=0.01,
                end_value=0.001,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


transform = DINOTransform(
    global_crop_size=224,
    local_crop_size=96,
    global_crop_scale=(0.3, 1),
    local_crop_scale=(0.05, 0.3),
    gaussian_blur=(0.5, 0.5, 0.5),
    n_local_views=6,
)
