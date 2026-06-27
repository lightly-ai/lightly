# Example: BEIT pre-training on CIFAR-10.
# This script demonstrates how to use the BEIT model for masked image
# modeling (MIM) pre-training on a small subset of CIFAR-10.

# NOTE: The ImageTokenizer must be pre-trained before BEIT pre-training.
# The tokenizer follows the DALL-E discrete VAE architecture and can be
# trained as follows:
#
#     tokenizer = ImageTokenizer(vocab_size=8192)
#     optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
#
#     for images in dataloader:
#         logits, recon = tokenizer(images)
#         loss = F.mse_loss(recon, images)  # Reconstruction loss
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#     torch.save(tokenizer.state_dict(), "tokenizer.pth")
#
# During BEIT pre-training, load and freeze the tokenizer:
#     tokenizer.load_state_dict(torch.load("tokenizer.pth"))
#     tokenizer.eval()
#     for param in tokenizer.parameters():
#         param.requires_grad = False


from __future__ import annotations

import torch
import torchvision
from torch import nn
from torch.utils.data import Subset

from lightly.loss import MaskedImageModelingLoss
from lightly.models.modules import BEITEncoder, ImageTokenizer, MIMHead
from lightly.transforms import BEITTransform


class BEIT(nn.Module):
    """Simple BEIT pre-training wrapper.

    Attributes:
        encoder:
            BEIT Vision Transformer encoder.
        tokenizer:
            Discrete visual tokenizer (e.g., dVAE or VQ-VAE).
        head:
            Linear projection head mapping patch features to vocabulary
            logits.
    """

    def __init__(
        self,
        encoder: BEITEncoder,
        tokenizer: ImageTokenizer,
    ) -> None:
        """Initializes the BEIT model.

        Args:
            encoder:
                BEIT encoder instance.
            tokenizer:
                Visual tokenizer for generating target token IDs.
        """
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.head = MIMHead(
            embed_dim=encoder.embed_dim,
            vocab_size=tokenizer.vocab_size,
        )

    def forward(
        self,
        images: torch.Tensor,
        bool_masked_pos: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for masked image modeling.

        Args:
            images:
                Input images of shape (B, C, H, W).
            bool_masked_pos:
                Boolean mask of shape (B, N) indicating masked patches.

        Returns:
            A tuple of:
                - mim_logits: Logits for masked positions.
                - token_targets: Target token IDs for masked positions.
        """
        enc_out = self.encoder(
            x=images,
            bool_masked_pos=bool_masked_pos,
        )
        patch_features = enc_out["patch_features"]

        all_logits = self.head(patch_features=patch_features)
        mim_logits = all_logits[bool_masked_pos]

        with torch.no_grad():
            token_ids = self.tokenizer.tokenize(x=images)
            token_targets = token_ids[bool_masked_pos]

        return mim_logits, token_targets


# Smaller encoder for fast experimentation: depth=2, embed_dim=256
encoder = BEITEncoder(
    img_size=224,
    patch_size=8,
    embed_dim=256,
    depth=2,
    num_heads=4,
)
tokenizer = ImageTokenizer(vocab_size=512)
model = BEIT(encoder=encoder, tokenizer=tokenizer)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device=device)

transform = BEITTransform(input_size=224, patch_size=8)

# CIFAR-10: downloads fast, small images
dataset = torchvision.datasets.CIFAR10(
    root="datasets/cifar10",
    train=True,
    download=True,
    transform=transform,
)

# Only use 200 samples for quick demonstration
dataset = Subset(dataset, indices=list(range(200)))

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

criterion = MaskedImageModelingLoss()
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=1.5e-3,
    weight_decay=0.05,
)

print("Starting Training")
model.train()
for epoch in range(2):
    total_loss = 0.0
    for batch in dataloader:
        images, _ = batch
        images = images.to(device=device)

        bool_masked_pos = transform.mask_generator(
            batch_size=images.shape[0],
        ).to(device=device)

        mim_logits, token_targets = model(
            images=images,
            bool_masked_pos=bool_masked_pos,
        )
        loss = criterion(mim_logits, token_targets)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
