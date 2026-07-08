# This example requires the following dependencies to be installed:
# pip install "lightly[timm]"

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
import pytorch_lightning as pl
import torch
import torchvision
from timm.models.vision_transformer import vit_small_patch16_224
from torch import nn

from lightly.models import utils
from lightly.models.modules import MaskedVisionTransformerTIMM, PixioDecoderTIMM
from lightly.transforms import MAETransform


class Pixio(pl.LightningModule):
    def __init__(self):
        super().__init__()

        decoder_dim = 512
        vit = vit_small_patch16_224(img_size=256, reg_tokens=7)
        self.mask_ratio = 0.75
        self.grid_size = 4
        self.patch_size = vit.patch_embed.patch_size[0]
        self.num_prefix_tokens = vit.num_prefix_tokens
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = PixioDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=32,
            decoder_num_heads=16,
            num_prefix_tokens=self.num_prefix_tokens,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
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
        idx_keep, idx_mask = utils.random_grid_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            grid_size=self.grid_size,
            num_prefix_tokens=self.num_prefix_tokens,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for the prefix tokens
        target = utils.get_at_index(patches, idx_mask - self.num_prefix_tokens)
        target = utils.normalize_mean_var(target)

        loss = self.criterion(x_pred, target)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        return optim


model = Pixio()

transform = MAETransform(input_size=256)


# we ignore object detection annotations by setting target_transform to return 0
def target_transform(t):
    return 0


dataset = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=target_transform,
)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(
    max_epochs=10,
    devices=1,
    accelerator=accelerator,
)
trainer.fit(model=model, train_dataloaders=dataloader)
