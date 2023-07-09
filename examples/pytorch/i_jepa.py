import torch
import torchvision
from torch import nn

from lightly.models import utils
from lightly.models.modules import i_jepa
from lightly.transforms.mae_transform import IJEPATransform


class I_JEPA(nn.Module):
    pass


vit_for_predictor = torchvision.models.vit_b_32(pretrained=False)
vit_for_embedder = torchvision.models.vit_b_32(pretrained=False)
model = I_JEPA(vit_for_predictor, vit_for_embedder)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = IJEPATransform()
# we ignore object detection annotations by setting target_transform to return 0
dataset = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=lambda t: 0,
)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")
