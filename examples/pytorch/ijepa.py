import copy

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from lightly.data.collate import IJEPAMaskCollator
from lightly.models import utils
from lightly.models.modules.ijepa import IJEPABackbone, IJEPAPredictor
from lightly.transforms.ijepa_transform import IJEPATransform


class IJEPA(nn.Module):
    def __init__(self, vit_encoder, vit_predictor, momentum_scheduler):
        super().__init__()
        self.encoder = IJEPABackbone.from_vit(vit_encoder)
        self.predictor = IJEPAPredictor.from_vit_encoder(
            vit_predictor.encoder,
            (vit_predictor.image_size // vit_predictor.patch_size) ** 2,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        self.momentum_scheduler = momentum_scheduler

    def forward_target(self, imgs, masks_enc, masks_pred):
        with torch.no_grad():
            h = self.target_encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            B = len(h)
            # -- create targets (masked regions of h)
            h = utils.apply_masks(h, masks_pred)
            h = utils.repeat_interleave_batch(h, B, repeat=len(masks_enc))
            return h

    def forward_context(self, imgs, masks_enc, masks_pred):
        z = self.encoder(imgs, masks_enc)
        z = self.predictor(z, masks_enc, masks_pred)
        return z

    def forward(self, imgs, masks_enc, masks_pred):
        z = self.forward_context(imgs, masks_enc, masks_pred)
        h = self.forward_target(imgs, masks_enc, masks_pred)
        return z, h

    def update_target_encoder(
        self,
    ):
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(
                self.encoder.parameters(), self.target_encoder.parameters()
            ):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)


collator = IJEPAMaskCollator(
    input_size=(224, 224),
    patch_size=32,
)

transform = IJEPATransform()

# we ignore object detection annotations by setting target_transform to return 0
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")
dataset = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=lambda t: 0,
)
data_loader = torch.utils.data.DataLoader(
    dataset, collate_fn=collator, batch_size=10, persistent_workers=False
)

ema = (0.996, 1.0)
ipe_scale = 1.0
ipe = len(data_loader)
num_epochs = 10
momentum_scheduler = (
    ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
    for i in range(int(ipe * num_epochs * ipe_scale) + 1)
)

vit_for_predictor = torchvision.models.vit_b_32(pretrained=False)
vit_for_embedder = torchvision.models.vit_b_32(pretrained=False)
model = IJEPA(vit_for_embedder, vit_for_predictor, momentum_scheduler)

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Starting Training")
for epoch in range(num_epochs):
    total_loss = 0
    for udata, masks_enc, masks_pred in tqdm(data_loader):

        def load_imgs():
            # -- unsupervised imgs
            imgs = udata[0].to(device, non_blocking=True)
            masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
            return (imgs, masks_1, masks_2)

        imgs, masks_enc, masks_pred = load_imgs()
        z, h = model(imgs, masks_enc, masks_pred)
        loss = criterion(z, h)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.update_target_encoder()

    avg_loss = total_loss / len(data_loader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
