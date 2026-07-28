from typing import List, Tuple

from pytorch_lightning import LightningModule
from timm.models.vision_transformer import vit_base_patch16_224
from torch import Tensor
from torch.nn import Linear, MSELoss
from torch.optim import AdamW

from lightly.models import utils
from lightly.models.modules import (
    MaskedVisionTransformerDecoderTIMM,
    MaskedVisionTransformerTIMM,
)
from lightly.transforms import MAETransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler


class MAE(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        decoder_dim = 512
        vit = vit_base_patch16_224()

        self.mask_ratio = 0.75
        self.patch_size = vit.patch_embed.patch_size[0]
        self.sequence_length = vit.patch_embed.num_patches + vit.num_prefix_tokens
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.decoder_embed = Linear(vit.embed_dim, decoder_dim)
        self.decoder = MaskedVisionTransformerDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            embed_dim=decoder_dim,
            depth=8,
            num_heads=16,
            num_prefix_tokens=vit.num_prefix_tokens,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        self.prediction_head = Linear(decoder_dim, self.patch_size**2 * 3)
        self.criterion = MSELoss()

        self.online_classifier = OnlineLinearClassifier(
            feature_dim=768, num_classes=num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(images=x)

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # embed encoded tokens into the decoder dimension
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder_embed(x_encoded)

        # scatter the encoded tokens into a full-length sequence; the decoder fills the
        # masked positions with the mask token
        x_masked = x_decode.new_zeros(
            batch_size, self.sequence_length, x_decode.shape[-1]
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder(x_masked, idx_mask=idx_mask)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.prediction_head(x_pred)
        return x_pred

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        images = images[0]  # images is a list containing only one view
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        features = self.forward_encoder(images, idx_keep)
        predictions = self.forward_decoder(features, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(predictions, target)
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        cls_features = features[:, 0]
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
            [self.backbone, self.decoder_embed, self.decoder, self.prediction_head]
        )
        optimizer = AdamW(
            [
                {"name": "mae", "params": params},
                {
                    "name": "mae_no_weight_decay",
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


transform = MAETransform()
