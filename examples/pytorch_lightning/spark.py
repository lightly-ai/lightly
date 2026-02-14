# This example requires the following dependencies to be installed:
# pip install lightly

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import timm
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from torch import nn
from torchvision.transforms import v2

from lightly.loss.sparse_spark import SparKPatchReconLoss

## The global projection head is the same as the Barlow Twins one
from lightly.models.modules.sparse_spark import (
    SparKDensifier,
    SparKMasker,
    SparKMaskingOuptut,
    SparKOutputDecoder,
    UNetDecoder,
    dense_model_to_sparse,
)
from lightly.models.utils import patchify


def get_downsample_ratio_from_timm_model(model: nn.Module) -> int:
    return model.feature_info[-1]["reduction"]


def get_enc_feat_map_chs_from_timm_model(model: nn.Module) -> list[int]:
    return [fi["num_chs"] for fi in model.feature_info]


class SparseSparK(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 416,
        mask_ratio: float = 0.6,
        densify_norm: str = "bn",
        sbn=False,
    ):
        super().__init__()
        backbone = timm.create_model(
            "resnet18", drop_path_rate=0.05, features_only=True
        )
        downsample_ratio = get_downsample_ratio_from_timm_model(backbone)
        enc_feat_map_chs = get_enc_feat_map_chs_from_timm_model(backbone)
        self.sparse_encoder = dense_model_to_sparse(backbone, sbn=sbn, verbose=True)
        self.fmap_h = input_size // downsample_ratio
        self.fmap_w = input_size // downsample_ratio
        self.dense_decoder = UNetDecoder(
            downsample_ratio,
            width=enc_feat_map_chs[-1],
        )
        self.masker = SparKMasker(
            feature_map_size=(self.fmap_h, self.fmap_w),
            downsample_ratio=downsample_ratio,
            mask_ratio=mask_ratio,
        )
        self.densifier = SparKDensifier(
            encoder_in_channels=enc_feat_map_chs,
            decoder_in_channel=self.dense_decoder.width,
            densify_norm_str=densify_norm.lower(),
            sbn=sbn,
        )
        self.downsample_ratio = downsample_ratio
        # loss module for patch reconstruction
        self.recon_loss_fn = SparKPatchReconLoss()
        # output decoder for visualization (pass minimal spatial props)
        self.output_decoder = SparKOutputDecoder(
            self.fmap_h,
            self.fmap_w,
            downsample_ratio,
        )

    def forward(
        self,
        inp_bchw: torch.Tensor,
        vis=False,
    ):
        # step1. Mask
        mask_out: SparKMaskingOuptut = self.masker(inp_bchw)
        masked_bchw, per_level_mask = mask_out
        active_b1fHfW = per_level_mask[0]
        active_b1hw = per_level_mask[-1]
        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcffs: list[torch.Tensor] = self.sparse_encoder(masked_bchw)
        # step3. Densify: get hierarchical dense features for decoding
        to_dec = self.densifier(fea_bcffs)
        # step4. Decode and reconstruct
        rec_bchw = self.dense_decoder(to_dec)
        inp, rec = (
            patchify(inp_bchw, self.downsample_ratio),
            patchify(rec_bchw, self.downsample_ratio),
        )  # inp and rec: (B, L = f*f, N = C*downsample_raito**2)

        recon_loss, mean, var = self.recon_loss_fn(inp, rec, active_b1fHfW)

        if vis:
            return self.output_decoder(rec, mean, var, inp_bchw, active_b1hw)
        else:
            return recon_loss

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
        return torch.optim.SGD(
            self.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4
        )


model = SparseSparK(input_size=416)


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
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    num_workers=8,
)


accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(
    max_epochs=30,
    devices=1,
    accelerator=accelerator,
    callbacks=[
        RichProgressBar(),
    ],
)
trainer.fit(
    model=model,
    train_dataloaders=dataloader,
)
