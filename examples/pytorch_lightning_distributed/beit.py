# This example requires the following dependencies to be installed:
# pip install "lightly[timm]"

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torchvision
from torch import nn

from lightly.loss import MaskedImageModelingLoss
from lightly.models.modules import BEITEncoder, ImageTokenizer, MIMHead
from lightly.transforms import BEITTransform


class BEIT(pl.LightningModule):
    """BEIT pre-training module for masked image modeling with DDP support.

    Attributes:
        backbone:
            BEIT Vision Transformer encoder.
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
        """Initializes the BEIT pre-training module.

        Args:
            img_size:
                Spatial resolution of input images.
            patch_size:
                Size of each patch.
            embed_dim:
                Dimension of token embeddings.
            depth:
                Number of Transformer blocks.
            num_heads:
                Number of attention heads per block.
            vocab_size:
                Size of the discrete visual vocabulary.
        """
        super().__init__()
        self.backbone = BEITEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.num_patches = self.backbone.num_patches
        self.projection_head = MIMHead(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
        )
        self.tokenizer = ImageTokenizer(vocab_size=vocab_size)
        self.criterion = MaskedImageModelingLoss()

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Performs a single training step.

        Args:
            batch:
                Tuple of (images, targets) where images is a tensor of
                shape (B, C, H, W).
            batch_idx:
                Index of the current batch.

        Returns:
            The computed loss value.
        """
        images, _ = batch
        batch_size = images.shape[0]

        # Generate blockwise mask
        bool_masked_pos = self.transform.mask_generator(
            batch_size=batch_size,
        ).to(device=images.device)

        # Forward pass through encoder + head
        enc_out = self.backbone(
            x=images,
            bool_masked_pos=bool_masked_pos,
        )
        patch_features = enc_out["patch_features"]
        all_logits = self.projection_head(patch_features=patch_features)

        # Select masked positions
        mim_logits = all_logits[bool_masked_pos]

        # Get token targets from frozen tokenizer
        with torch.no_grad():
            token_ids = self.tokenizer.tokenize(x=images)
            token_targets = token_ids[bool_masked_pos]

        loss = self.criterion(mim_logits, token_targets)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer for training.

        Returns:
            AdamW optimizer.
        """
        optim = torch.optim.AdamW(
            params=self.parameters(),
            lr=1.5e-3,
            weight_decay=0.05,
        )
        return optim


model = BEIT()

transform = BEITTransform(input_size=224, patch_size=8)
model.transform = transform  # Store transform for access in training_step


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
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

# DDP training: uses 2 GPUs, synchronized batch norm
trainer = pl.Trainer(
    max_epochs=10,
    devices=2,
    accelerator=accelerator,
    strategy="ddp",
    sync_batchnorm=True,
)
trainer.fit(model=model, train_dataloaders=dataloader)
