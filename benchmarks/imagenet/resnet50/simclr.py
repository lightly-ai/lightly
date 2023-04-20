from typing import List, Tuple

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Sequential
from torchvision.models import resnet50

from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms import SimCLRTransform
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler


class SimCLR(LightningModule):
    def __init__(self, batch_size: int, epochs: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.epochs = epochs

        # Resnet50 backbone without classification head.
        self.backbone = Sequential(*list(resnet50().children())[:-1])
        self.projection_head = SimCLRProjectionHead()
        self.criterion = NTXentLoss(temperature=0.1, gather_distributed=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        (view0, view1), _, _ = batch
        z0 = self.projection_head(self.forward(view0).flatten(start_dim=1))
        z1 = self.projection_head(self.forward(view1).flatten(start_dim=1))
        loss = self.criterion(z0, z1)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = LARS(
            self.parameters(),
            lr=0.3
            * self.batch_size
            * self.trainer.num_devices
            * self.trainer.num_nodes
            / 256,
            weight_decay=1e-6,
        )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=10,
            max_epochs=self.epochs,
        )
        return [optimizer], [scheduler]


transform = SimCLRTransform()
