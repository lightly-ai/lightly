# This example requires the following dependencies to be installed:
# pip install lightly[timm]

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torchvision
from timm.models.vision_transformer import vit_small_patch16_224
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW

from lightly.loss import LeJEPALoss
from lightly.models.modules import LeJEPAProjectionHead
from lightly.transforms.dino_transform import DINOTransform


def _get_backbone_output_dim(backbone: Module) -> int:
    """Get the output dimension of a backbone by passing a dummy input through it."""
    with torch.inference_mode():
        dummy_input = torch.zeros(1, 3, 224, 224)
        output = backbone(dummy_input)
        output_dim = output.shape[1]
    return output_dim


class LeJEPA(pl.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.backbone = vit_small_patch16_224(
            pretrained=False,
            pos_embed="learn",
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=0.1,
        )

        backbone_out_dims = _get_backbone_output_dim(self.backbone)
        self.projection_head = LeJEPAProjectionHead(input_dim=backbone_out_dims)
        self.criterion = LeJEPALoss()

    def forward(self, x: Tensor) -> Tensor:
        emb = self.backbone(x)
        proj = self.projection_head(emb)
        return proj

    def training_step(
        self, batch: tuple[list[Tensor], Tensor], batch_idx: int
    ) -> Tensor:
        views = batch[0]
        global_views = views[:2]
        local_views = views[2:]

        global_proj = torch.stack([self(view) for view in global_views])
        local_proj = torch.stack([self(view) for view in local_views])

        loss = self.criterion(local_proj=local_proj, global_proj=global_proj)
        return loss

    def configure_optimizers(self) -> AdamW:
        optim = AdamW(self.parameters(), lr=5e-4, weight_decay=5e-2)
        return optim


model = LeJEPA()

transform = DINOTransform(
    global_crop_scale=(0.3, 1),
    local_crop_scale=(0.05, 0.3),
    gaussian_blur=(0.5, 0.5, 0.5),
    n_local_views=6,
)

dataset = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc",
    download=True,
    transform=transform,
)
# Or create a dataset from a folder containing images or videos.
# dataset = LightlyDataset("path/to/folder")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(max_epochs=50, devices=1, accelerator=accelerator)
trainer.fit(model=model, train_dataloaders=dataloader)
