import sys

import torch
import torchvision
from torch import nn

from lightly.models import utils
from lightly.models.modules import masked_vision_transformer_timm
from lightly.transforms.mae_transform import MAETransform  # Same transform as MAE

try:
    from timm.models import vision_transformer
except ImportError:
    print("TIMM is not available. Please install in order to run this example.")
    sys.exit(1)


class SimMIM(nn.Module):
    def __init__(self, vit):
        super().__init__()

        decoder_dim = vit.embed_dim
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_embed.patch_size[0]
        self.sequence_length = vit.patch_embed.num_patches + vit.num_prefix_tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # same backbone as MAE
        self.backbone = masked_vision_transformer_timm.MaskedVisionTransformerTIMM(
            vit=vit, mask_token=self.mask_token
        )

        # the decoder is a simple linear layer
        self.patch_size = vit.patch_embed.patch_size[0]
        self.decoder = nn.Linear(vit.embed_dim, vit.patch_embed.patch_size[0] ** 2 * 3)

    def forward_encoder(self, images, batch_size, idx_mask):
        # pass all the tokens to the encoder, both masked and non masked ones
        # tokens = self.backbone.images_to_tokens(images, prepend_class_token=True)
        # tokens_masked = utils.mask_at_index(tokens, idx_mask, self.mask_token)
        # return self.backbone.encoder(tokens_masked)
        return self.backbone.encode(images=images, idx_mask=idx_mask)

    def forward_decoder(self, x_encoded):
        return self.decoder(x_encoded)

    def forward(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        # Encoding...
        x_encoded = self.forward_encoder(images, batch_size, idx_mask)
        x_encoded_masked = utils.get_at_index(x_encoded, idx_mask)

        # Decoding...
        x_out = self.forward_decoder(x_encoded_masked)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)

        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        return x_out, target


device = "cuda" if torch.cuda.is_available() else "cpu"

vit = vision_transformer.vit_base_patch32_224()
vit = vit.to(device)
model = SimMIM(vit)


model.to(device)

transform = MAETransform()
# we ignore object detection annotations by setting target_transform to return 0
dataset = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=lambda t: 0,
)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

# L1 loss as paper suggestion
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        views = batch[0]
        images = views[0].to(device)  # views contains only a single view
        predictions, targets = model(images)

        loss = criterion(predictions, targets)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
