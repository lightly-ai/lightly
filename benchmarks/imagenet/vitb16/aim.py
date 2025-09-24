import sys
sys.path.append('/git/lightly')
from typing import List, Optional, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from lightly.models import utils
from lightly.models.modules import AIMPredictionHead, MaskedCausalVisionTransformer
from lightly.models.utils import get_2d_sincos_pos_embed, random_prefix_mask
from lightly.transforms import AIMTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler


class AIM(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        vit = MaskedCausalVisionTransformer(
            img_size=224,
            patch_size=14,
            num_classes=num_classes,
            embed_dim=1536,
            depth=24,
            num_heads=12,
            qk_norm=False,
            class_token=False,
            no_embed_class=True,
        )
        utils.initialize_2d_sine_cosine_positional_embedding(
            pos_embedding=vit.pos_embed, has_class_token=vit.has_class_token
        )
        self.patch_size = vit.patch_embed.patch_size[0]
        self.num_patches = vit.patch_embed.num_patches

        self.backbone = vit
        self.projection_head = AIMPredictionHead(
            input_dim=vit.embed_dim, output_dim=3 * self.patch_size**2
        )

        self.criterion = MSELoss()

        self.online_classifier = OnlineLinearClassifier(
            feature_dim=vit.embed_dim, num_classes=num_classes
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        features = self.backbone.forward_features(x, mask=mask)
        # TODO: We use mean aggregation for simplicity. The paper uses
        # AttentionPoolingClassifier to get the class features. But this is not great
        # as it requires training an additional head.
        # https://github.com/apple/ml-aim/blob/1eaedecc4d584f2eb7c6921212d86a3a694442e1/aim/torch/layers.py#L337
        return features.mean(dim=1).flatten(start_dim=1)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        views, targets = batch[0], batch[1]
        images = views[0]  # AIM has only a single view
        batch_size = images.shape[0]

        mask = random_prefix_mask(
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

        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        # TODO: We could use AttentionPoolingClassifier instead of mean aggregation:
        # https://github.com/apple/ml-aim/blob/1eaedecc4d584f2eb7c6921212d86a3a694442e1/aim/torch/layers.py#L337
        cls_features = features.mean(dim=1).flatten(start_dim=1)
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
            [self.backbone, self.projection_head]
        )
        optimizer = AdamW(
            [
                {"name": "aim", "params": params},
                {
                    "name": "aim_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.001 * self.batch_size_per_device * self.trainer.world_size / 4096,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=31250 / 125000 * self.trainer.estimated_stepping_batches,
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Union[int, float, None] = None,
        gradient_clip_algorithm: Union[str, None] = None,
    ) -> None:
        self.clip_gradients(
            optimizer=optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )


transform = AIMTransform()
