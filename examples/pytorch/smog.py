# This example requires the following dependencies to be installed:
# pip install lightly
# pip install "scikit-learn>=1.7.1"

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy

import torch
import torchvision
from sklearn.cluster import KMeans
from torch import nn

from lightly.models import utils
from lightly.models.modules.heads import (
    SMoGPredictionHead,
    SMoGProjectionHead,
    SMoGPrototypes,
)
from lightly.models.modules.memory_bank import MemoryBankModule
from lightly.transforms.smog_transform import SMoGTransform


class SMoGModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = SMoGProjectionHead(512, 2048, 128)
        self.prediction_head = SMoGPredictionHead(128, 2048, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        self.n_groups = 300
        self.smog = SMoGPrototypes(
            group_features=torch.rand(self.n_groups, 128), beta=0.99
        )

    def _cluster_features(self, features: torch.Tensor) -> torch.Tensor:
        # clusters the features using sklearn
        # (note: faiss is probably more efficient)
        features = features.cpu().numpy()
        kmeans = KMeans(self.n_groups).fit(features)
        clustered = torch.from_numpy(kmeans.cluster_centers_).float()
        clustered = torch.nn.functional.normalize(clustered, dim=1)
        return clustered

    def reset_group_features(self, memory_bank):
        # see https://arxiv.org/pdf/2207.06167.pdf Table 7b)
        features = memory_bank.bank
        group_features = self._cluster_features(features)
        self.smog.set_group_features(group_features)

    def reset_momentum_weights(self):
        # see https://arxiv.org/pdf/2207.06167.pdf Table 7b)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        encoded = self.projection_head(features)
        predicted = self.prediction_head(encoded)
        return encoded, predicted

    def forward_momentum(self, x):
        features = self.backbone_momentum(x).flatten(start_dim=1)
        encoded = self.projection_head_momentum(features)
        return encoded


batch_size = 256

resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SMoGModel(backbone)

# memory bank because we reset the group features every 300 iterations
memory_bank_size = 300 * batch_size
memory_bank = MemoryBankModule(size=(memory_bank_size, 128))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = SMoGTransform(
    crop_sizes=(32, 32),
    crop_counts=(1, 1),
    gaussian_blur_probs=(0.0, 0.0),
    crop_min_scales=(0.2, 0.2),
    crop_max_scales=(1.0, 1.0),
)
dataset = torchvision.datasets.CIFAR10(
    "datasets/cifar10", download=True, transform=transform
)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder", transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6
)

global_step = 0

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        (x0, x1) = batch[0]

        if batch_idx % 2:
            # swap batches every second iteration
            x1, x0 = x0, x1

        x0 = x0.to(device)
        x1 = x1.to(device)

        if global_step > 0 and global_step % 300 == 0:
            # reset group features and weights every 300 iterations
            model.reset_group_features(memory_bank=memory_bank)
            model.reset_momentum_weights()
        else:
            # update momentum
            utils.update_momentum(model.backbone, model.backbone_momentum, 0.99)
            utils.update_momentum(
                model.projection_head, model.projection_head_momentum, 0.99
            )

        x0_encoded, x0_predicted = model(x0)
        x1_encoded = model.forward_momentum(x1)

        # update group features and get group assignments
        assignments = model.smog.assign_groups(x1_encoded)
        group_features = model.smog.get_updated_group_features(x0_encoded)
        logits = model.smog(x0_predicted, group_features, temperature=0.1)
        model.smog.set_group_features(group_features)

        loss = criterion(logits, assignments)

        # use memory bank to periodically reset the group features with k-means
        memory_bank(x0_encoded, update=True)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        total_loss += loss.detach()

    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
