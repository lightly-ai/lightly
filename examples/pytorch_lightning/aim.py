# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.models import utils
from lightly.models.modules import AIMPredictionHead, MaskedCausalVisionTransformer
from lightly.transforms import AIMTransform


class AIM(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        vit = MaskedCausalVisionTransformer(
            img_size=224,
            patch_size=32,
            embed_dim=768,
            depth=12,
            num_heads=12,
            qk_norm=False,
            class_token=False,
            no_embed_class=True,
        )
        utils.initialize_2d_sine_cosine_positional_embedding(
            pos_embedding=vit.pos_embed
        )
        self.patch_size = vit.patch_embed.patch_size[0]
        self.num_patches = vit.patch_embed.num_patches

        self.backbone = vit
        self.projection_head = AIMPredictionHead(
            input_dim=vit.embed_dim, output_dim=3 * self.patch_size**2, num_blocks=1
        )

        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        images, targets = batch[0], batch[1]
        images = images[0]  # images is a list containing only one view
        batch_size = images.shape[0]

        mask = utils.random_prefix_mask(
            size=(batch_size, self.num_patches),
            max_prefix_length=self.num_patches - 1,
            device=images.device,
        )
        features = self.backbone.forward_features(images, mask=mask)
        # Add positional embedding before head.
        features = self.backbone._pos_embed(features)
        predictions = self.projection_head(features)

        # Convert images to patches and normalize them.
        patches = utils.patchify(images, self.patch_size)
        patches = utils.normalize_mean_var(patches, dim=-1)

        loss = self.criterion(predictions, patches)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        return optim


model = AIM()

transform = AIMTransform()
# we ignore object detection annotations by setting target_transform to return 0
dataset = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=lambda t: 0,
)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(max_epochs=10, devices=1, accelerator=accelerator)
trainer.fit(model=model, train_dataloaders=dataloader)
