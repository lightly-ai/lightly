import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import copy

from lightly.models import utils
from lightly.models.modules import i_jepa
from lightly.transforms.mae_transform import IJEPATransform
from lightly.data.collate import IJEPAMaskCollator


class I_JEPA(nn.Module):
    def __init__(self, vit_encoder, vit_predictor, momentum_scheduler):
        super().__init__()
        self.encoder = i_jepa.IJEPA_encoder.from_vit_encoder(vit_encoder)
        self.predictor = i_jepa.IJEPA_predictor.from_vit_encoder(vit_predictor)
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
        z = self.forward_context(self, imgs, masks_enc, masks_pred)
        h = self.forward_target(self, imgs, masks_enc, masks_pred)
        return z, h
    
    def update_target_encoder(self,):
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)


vit_for_predictor = torchvision.models.vit_b_32(pretrained=False)
vit_for_embedder = torchvision.models.vit_b_32(pretrained=False)
model = I_JEPA(vit_for_predictor, vit_for_embedder)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

collator = IJEPAMaskCollator(
    input_size=(224,224),
    patch_size=32,
)

transform = IJEPATransform()
dataset = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=lambda t: 0,
)
data_loader = torch.utils.data.DataLoader(
    dataset,
    collate_fn=collator,
    batch_size=1,
    persistent_workers=False
)

# we ignore object detection annotations by setting target_transform to return 0
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")
print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for itr, (udata, masks_enc, masks_pred) in enumerate(data_loader):

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
    avg_loss = total_loss / len(data_loader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")  