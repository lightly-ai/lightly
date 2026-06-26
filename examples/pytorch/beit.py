import torch
import torchvision
from torch import nn
from torch.utils.data import Subset

from lightly.loss import MaskedImageModelingLoss
from lightly.models.modules import BEITEncoder, ImageTokenizer, MIMHead
from lightly.transforms import BEITTransform


class BEIT(nn.Module):
    def __init__(self, encoder: BEITEncoder, tokenizer: ImageTokenizer) -> None:
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
        patch_features = self.encoder(images, bool_masked_pos=bool_masked_pos)[
            "patch_features"
        ]
        all_logits = self.head(patch_features)
        mim_logits = all_logits[bool_masked_pos]
        with torch.no_grad():
            token_ids = self.tokenizer.tokenize(images)
            token_targets = token_ids[bool_masked_pos]
        return mim_logits, token_targets


# smaller encoder: depth=2, embed_dim=256
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
model.to(device)

transform = BEITTransform(input_size=224, patch_size=8)


# CIFAR-10: downloads fast, small images
dataset = torchvision.datasets.CIFAR10(
    root="datasets/cifar10",
    train=True,
    download=True,
    transform=transform,
)

# only use 200 samples
dataset = Subset(dataset, indices=list(range(200)))

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)


criterion = MaskedImageModelingLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.05)

print("Starting Training")
for epoch in range(2):
    total_loss = 0
    for batch in dataloader:
        images, _ = batch
        images = images.to(device)
        bool_masked_pos = transform.mask_generator(batch_size=images.shape[0]).to(
            device
        )
        mim_logits, token_targets = model(images, bool_masked_pos)
        loss = criterion(mim_logits, token_targets)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
