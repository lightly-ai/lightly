# This example requires the following dependencies to be installed:
# pip install "lightly[timm]"

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torchvision

from lightly.loss import MaskedImageModelingLoss
from lightly.models.modules import BEITEncoder, BEiTImageTokenizer, BEiTMIMHead
from lightly.transforms import BEiTTransform


class BEiT(pl.LightningModule):
    """BEiT pre-training module for masked image modeling with DDP support.

    Attributes:
        backbone:
            BEiT Vision Transformer encoder.
        projection_head:
            MIM head mapping patch features to vocabulary logits.
        tokenizer:
            Discrete visual tokenizer for generating target token IDs.
        criterion:
            Cross-entropy loss for masked token prediction.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 8,
        embed_dim: int = 256,
        depth: int = 2,
        num_heads: int = 4,
        vocab_size: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = BEITEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.num_patches = self.backbone.num_patches
        self.projection_head = BEiTMIMHead(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
        )
        self.tokenizer = BEiTImageTokenizer(vocab_size=vocab_size)
        self.tokenizer.requires_grad_(False)
        self.tokenizer.eval()
        self.criterion = MaskedImageModelingLoss()

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        images, _ = batch
        batch_size = images.shape[0]

        bool_masked_pos = self.transform.mask_generator(
            batch_size=batch_size,
        ).to(device=images.device)

        enc_out = self.backbone(
            x=images,
            bool_masked_pos=bool_masked_pos,
        )
        patch_features = enc_out["patch_features"]
        all_logits = self.projection_head(patch_features=patch_features)

        mim_logits = all_logits[bool_masked_pos]

        with torch.no_grad():
            token_ids = self.tokenizer.tokenize(x=images)
            token_targets = token_ids[bool_masked_pos]

        loss = self.criterion(mim_logits, token_targets)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim = torch.optim.AdamW(
            params=self.parameters(),
            lr=1.5e-3,
            weight_decay=0.05,
        )
        return optim


model = BEiT()

transform = BEiTTransform(input_size=224, patch_size=8)
model.transform = transform  # Store transform for access in training_step

# Fast
dataset = torchvision.datasets.FakeData(
    size=200,
    image_size=(3, 224, 224),
    num_classes=20,
    transform=transform,
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
n_gpus = torch.cuda.device_count()


trainer = pl.Trainer(
    max_epochs=10,
    devices="auto",
    accelerator="gpu",
    sync_batchnorm=True,
    use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
)


trainer.fit(model=model, train_dataloaders=dataloader)
