from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from timm.models.vision_transformer import vit_base_patch16_224
from torch import Tensor
from torch.nn import MSELoss, Parameter
from torch.optim import AdamW

from lightly.models import utils
from lightly.models.modules import MaskedVisionTransformerTIMM, PixioDecoderTIMM
from lightly.transforms import MAETransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler


class PIXIO(LightningModule):
    """PIXIO pre-training [0].

    PIXIO is MAE with three changes: a deeper decoder, larger (grid/block) masking
    granularity, and multiple class tokens whose mean is used as the global
    representation. Implemented from the paper, not the reference code.

    The paper's headline config is used here (256px input, patch 16, 4x4 grid mask,
    8 class tokens, 512x32 decoder). The dense-prediction-optimal ablation uses a 2x2
    grid and 4 class tokens.

    - [0]: In Pursuit of Pixel Supervision for Visual Pre-training, 2025,
      https://arxiv.org/abs/2512.15715
    """

    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        decoder_dim = 512
        self.mask_ratio = 0.75
        self.grid_size = 4
        # vit-b/16 at 256px with 8 prefix tokens (1 cls + 7 reg). dynamic_img_size
        # lets downstream evaluation run at other resolutions via pos-embed resampling.
        vit = vit_base_patch16_224(img_size=256, reg_tokens=7, dynamic_img_size=True)
        self.num_prefix_tokens = vit.num_prefix_tokens
        self.patch_size = vit.patch_embed.patch_size[0]
        self.sequence_length = vit.patch_embed.num_patches + vit.num_prefix_tokens
        mask_token = Parameter(torch.zeros(1, 1, decoder_dim))
        torch.nn.init.normal_(mask_token, std=0.02)
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
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
            mask_token=mask_token,
        )
        self.criterion = MSELoss()

        self.online_classifier = OnlineLinearClassifier(
            feature_dim=vit.embed_dim, num_classes=num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        # Global representation = mean over the prefix (class) tokens.
        features = self.backbone.encode(images=x)
        return features[:, : self.num_prefix_tokens].mean(dim=1)

    def forward_encoder(self, images: Tensor, idx_keep: Tensor) -> Tensor:
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(
        self, x_encoded: Tensor, idx_keep: Tensor, idx_mask: Tensor
    ) -> Tensor:
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

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        images = images[0]  # images is a list containing only one view
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_grid_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            grid_size=self.grid_size,
            num_prefix_tokens=self.num_prefix_tokens,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        predictions = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # Reconstruction target: normalized pixel values of the masked patches.
        patches = utils.patchify(images, self.patch_size)
        target = utils.get_at_index(patches, idx_mask - self.num_prefix_tokens)
        target = utils.normalize_mean_var(target)

        loss = self.criterion(predictions, target)
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        cls_features = x_encoded[:, : self.num_prefix_tokens].mean(dim=1)
        cls_loss, cls_log = self.online_classifier.training_step(
            (cls_features.detach(), targets), batch_idx
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
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = utils.get_weight_decay_parameters(
            [self.backbone, self.decoder]
        )
        optimizer = AdamW(
            [
                {"name": "pixio", "params": params},
                {
                    "name": "pixio_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=1.5e-4 * self.batch_size_per_device * self.trainer.world_size / 256,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 40
                ),
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


transform = MAETransform(input_size=256)
