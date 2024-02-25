# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torchvision
from timm.models.vision_transformer import vit_base_patch32_224
from torch import nn

from lightly.models import utils
from lightly.models.modules import (
    masked_autoencoder_timm,
    masked_vision_transformer_timm,
)
from lightly.transforms.mae_transform import MAETransform


class MAE(pl.LightningModule):
    def __init__(self):
        super().__init__()

        decoder_dim = 512
        vit = vit_base_patch32_224()
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_embed.patch_size[0]
        mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.backbone = masked_vision_transformer_timm.MaskedVisionTransformerTIMM(
            vit=vit
        )
        self.sequence_length = self.backbone.sequence_length
        self.decoder = masked_autoencoder_timm.MAEDecoder(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
            mask_token=mask_token,
        )
        self.criterion = nn.MSELoss()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        views = batch[0]
        images = views[0]  # views contains only a single view
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        return optim


model = MAE()

transform = MAETransform()
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

# Train with DDP on multiple gpus. Distributed sampling is also enabled with
# replace_sampler_ddp=True.
trainer = pl.Trainer(
    max_epochs=10,
    devices="auto",
    accelerator="gpu",
    strategy="ddp",
    use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
)
trainer.fit(model=model, train_dataloaders=dataloader)
