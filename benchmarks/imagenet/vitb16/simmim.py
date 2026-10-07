"""SimMIM ImageNet benchmark.

SimMIM: A Simple Framework for Masked Image Modeling, 2021.
https://arxiv.org/abs/2111.09886
"""

from __future__ import annotations

from pytorch_lightning import LightningModule
from timm.models.vision_transformer import vit_base_patch16_224
from torch import Tensor, nn
from torch.optim import AdamW

from lightly.models import utils
from lightly.models.modules import MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler


class SimMIM(LightningModule):
    """SimMIM masked image modeling with a ViT-B/16 backbone.

    Masked patches are replaced by a learnable mask token, the full token
    sequence is encoded, and a single linear layer predicts the raw pixel values
    of the masked patches, which are trained with an L1 reconstruction loss.
    """

    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        vit = vit_base_patch16_224()
        self.mask_ratio = 0.6
        self.patch_size = vit.patch_embed.patch_size[0]
        self.num_prefix_tokens = vit.num_prefix_tokens
        self.sequence_length = vit.patch_embed.num_patches + self.num_prefix_tokens
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        # SimMIM uses a lightweight one-layer linear prediction head.
        self.decoder = nn.Linear(vit.embed_dim, self.patch_size**2 * 3)
        self.criterion = nn.L1Loss()

        self.online_classifier = OnlineLinearClassifier(
            feature_dim=vit.embed_dim, num_classes=num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(images=x)

    def training_step(
        self, batch: tuple[list[Tensor], Tensor, list[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        images = images[0]  # images is a list containing only one view
        batch_size = images.shape[0]
        # SimMIM masks 32x32 image regions, i.e. 2x2 blocks of 16x16 ViT patches.
        _, idx_mask = utils.random_grid_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            grid_size=2,
            num_prefix_tokens=self.num_prefix_tokens,
            device=images.device,
        )
        # Encode all tokens; masked positions are replaced by the mask token.
        x_encoded = self.backbone.encode(images=images, idx_mask=idx_mask)
        # Predict pixel values for the masked tokens.
        x_pred = self.decoder(utils.get_at_index(x_encoded, idx_mask))

        # Get image patches for the masked tokens, adjusting idx_mask for the
        # missing class token.
        patches = utils.patchify(images, self.patch_size)
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        cls_features = x_encoded[:, 0]
        cls_loss, cls_log = self.online_classifier.training_step(
            (cls_features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, list[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        cls_features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (cls_features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and the
        # classification head to improve performance.
        params, params_no_weight_decay = utils.get_weight_decay_parameters(
            [self.backbone, self.decoder]
        )
        optimizer = AdamW(
            [
                {"name": "simmim", "params": params},
                {
                    "name": "simmim_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # SimMIM uses base lr 8e-4 at batch size 2048, i.e. 1e-4 per 256 samples.
            lr=1e-4 * self.batch_size_per_device * self.trainer.world_size / 256,
            weight_decay=0.05,
            betas=(0.9, 0.999),
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


transform = MAETransform(min_scale=0.67)
