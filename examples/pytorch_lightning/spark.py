# This example requires the following dependencies to be installed:
# pip install "lightly[timm]"

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import timm
import torch
import torchvision
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import v2

import lightly.models.utils as model_utils
from lightly.loss.sparse_spark import SparKPatchReconLoss
from lightly.models.modules import sparse_spark

## The global projection head is the same as the Barlow Twins one
from lightly.models.modules.sparse_spark import (
    SparKDensifier,
    SparKMasker,
    SparKMaskingOutput,
    SparKOutputDecoder,
    UNetDecoder,
    sparse_layer_context,
)


def _get_downsample_ratio_from_timm_model(model: Module) -> int:
    if not hasattr(model, "feature_info"):
        raise ValueError(
            "The provided model does not have the required 'feature_info' attribute."
        )
    return model.feature_info[-1]["reduction"]


def _get_enc_feat_map_chs_from_timm_model(model: Module) -> list[int]:
    if not hasattr(model, "feature_info"):
        raise ValueError(
            "The provided model does not have the required 'feature_info' attribute."
        )
    return [fi["num_chs"] for fi in model.feature_info]


class SparseSparK(LightningModule):
    def __init__(
        self,
        input_size: int = 416,
        mask_ratio: float = 0.6,
        densify_norm: str = "bn",
        sbn=False,
    ):
        super().__init__()
        backbone = timm.create_model(
            model_name="resnet18", drop_path_rate=0.05, features_only=True
        )
        downsample_ratio = _get_downsample_ratio_from_timm_model(backbone)
        enc_feat_map_chs = _get_enc_feat_map_chs_from_timm_model(backbone)
        self.sparse_encoder = sparse_spark.dense_model_to_sparse(
            m=backbone, sbn=sbn, verbose=True
        )
        self.fmap_h = input_size // downsample_ratio
        self.fmap_w = input_size // downsample_ratio
        self.dense_decoder = UNetDecoder(
            up_sample_ratio=downsample_ratio,
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
            fmap_h=self.fmap_h,
            fmap_w=self.fmap_w,
            downsample_ratio=downsample_ratio,
        )

    def forward(
        self,
        inp_bchw: Tensor,
        vis=False,
    ):
        # step1. Mask
        mask_out: SparKMaskingOutput = self.masker(inp_bchw)
        masked_bchw, per_level_mask = mask_out
        active_b1fHfW = per_level_mask[0]
        active_b1hw = per_level_mask[-1]
        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        # Use sparse_layer_context to provide the mask to the sparse encoder and densifier.
        with sparse_layer_context(active_mask=active_b1fHfW):
            fea_bcffs: list[Tensor] = self.sparse_encoder(masked_bchw)
            # step3. Densify: get hierarchical dense features for decoding
            to_dec = self.densifier(fea_bcffs)
        # step4. Decode and reconstruct
        rec_bchw = self.dense_decoder(to_dec)
        inp, rec = (
            model_utils.patchify(inp_bchw, self.downsample_ratio),
            model_utils.patchify(rec_bchw, self.downsample_ratio),
        )  # inp and rec: (B, L = f*f, N = C*downsample_ratio**2)

        recon_loss, mean, var = self.recon_loss_fn(
            inp_patches=inp, rec_patches=rec, active_mask=active_b1fHfW
        )

        if vis:
            return self.output_decoder(
                rec_patches=rec,
                mean=mean,
                var=var,
                inp_bchw=inp_bchw,
                active_mask_full=active_b1hw,
            )
        else:
            return recon_loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> Tensor:
        img, _ = batch
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

trainer = pl.Trainer(
    max_epochs=30,
)

trainer.fit(
    model=model,
    train_dataloaders=dataloader,
)
