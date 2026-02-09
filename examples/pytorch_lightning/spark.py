# This example requires the following dependencies to be installed:
# pip install lightly

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import timm
import torch
import torchvision
from pytorch_lightning.callbacks import RichProgressBar
from torchvision.transforms import v2

## The global projection head is the same as the Barlow Twins one
from lightly.models.modules.sparse_spark import LightDecoder, SparK, SparseEncoder


class SparseSpark(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        backbone = timm.create_model(
            "resnet18", drop_path_rate=0.05, features_only=True
        )
        self.sparse_encoder = SparseEncoder(
            backbone, input_size=416, sbn=False, verbose=True
        )
        self.dense_decoder = LightDecoder(
            self.sparse_encoder.downsample_ratio,
            width=self.sparse_encoder.enc_feat_map_chs[-1],
        )
        self.spark = SparK(
            sparse_encoder=self.sparse_encoder,
            dense_decoder=self.dense_decoder,
        )

    def forward(self, x):
        return self.spark(x)

    def training_step(self, batch, batch_index) -> torch.Tensor:
        img, target = batch
        recon_loss = self.forward(img)
        # Log the training loss to logger and progress bar (per-step and per-epoch)
        self.log(
            "train_loss",
            recon_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return recon_loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return optim


model = SparseSpark()


# we ignore object detection annotations by setting target_transform to return 0
def target_transform(t):
    return 0


dataset = torchvision.datasets.Caltech101(
    "datasets/caltech101",
    download=True,
    transform=v2.Compose(
        [
            v2.Resize((416, 416)),
            v2.RGB(),
            v2.ToTensor(),
        ]
    ),
    target_transform=target_transform,
)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(
    max_epochs=10, devices=1, accelerator=accelerator, callbacks=[RichProgressBar()]
)
trainer.fit(model=model, train_dataloaders=dataloader)
