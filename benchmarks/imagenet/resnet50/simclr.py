from typing import List, Tuple

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torchvision.models import resnet50

from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms import SimCLRTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler


class SimCLR(LightningModule):
    def __init__(self, batch_size: int, epochs: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.epochs = epochs

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = SimCLRProjectionHead()
        self.criterion = NTXentLoss(temperature=0.1, gather_distributed=True)

        self.online_classifier = OnlineLinearClassifier()

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        (view0, view1), targets = batch[0], batch[1]
        features0 = self.forward(view0).flatten(start_dim=1)
        features1 = self.forward(view1).flatten(start_dim=1)
        z0 = self.projection_head(features0)
        z1 = self.projection_head(features1)
        loss = self.criterion(z0, z1)
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        cls_loss, cls_log = self.online_classifier.training_step(
            (features0, targets), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features, targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        parameters = list(self.backbone.parameters()) + list(
            self.projection_head.parameters()
        )
        optimizer = LARS(
            [
                {"params": parameters},
                {
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.3 * self.batch_size * self.trainer.world_size / 256,
            weight_decay=1e-6,
        )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=10,
            max_epochs=self.epochs,
        )
        return [optimizer], [scheduler]


transform = SimCLRTransform()
