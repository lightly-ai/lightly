# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy
import torch
from torch import nn
import torchvision
from sklearn.cluster import KMeans


from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.modules import SMoGProjectionHead
from lightly.models.modules import SMoGPredictionHead
from lightly.models.modules import SMoGPrototypes
from lightly.models import utils


class SMoGModel(nn.Module):
    def __init__(self, backbone, n_groups):
        super().__init__()

        self.backbone = backbone

        # create a model based on ResNet
        self.projection_head = SMoGProjectionHead(512, 2048, 128)
        self.prediction_head = SMoGPredictionHead(128, 2048, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        self.n_groups = n_groups  # 2% malus vs optimal setting of 3000 groups

        # create our loss
        group_features = torch.nn.functional.normalize(
            torch.rand(self.n_groups, 128), dim=1
        ).to(self.device)
        self.smog = SMoGPrototypes(group_features=group_features, beta=0.99)

    def _reset_group_features(self, memory_bank):
        # see Table 7b)
        features = self.memory_bank.bank
        if features is not None:
            features = features.t().cpu().numpy()
            kmeans = KMeans(self.n_groups).fit(features)
            new_features = torch.from_numpy(kmeans.cluster_centers_).float()
            new_features = torch.nn.functional.normalize(new_features, dim=1)
            self.smog.group_features = new_features.cuda()

    def _reset_momentum_weights(self):
        # see Table 7b)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)


batch_size = 256

resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SMoGModel(backbone)

# smog
memory_bank_size = (
    300 * batch_size
)  # because we reset the group features every 300 iterations
memory_bank = MemoryBankModule(size=memory_bank_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
dataset = LightlyDataset.from_torch_dataset(cifar10)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = SimCLRCollateFunction(input_size=32)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

global_step = 0

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        (x0, x1), _, _ = batch
        if batch_idx % 2:
            tmp = x1
            x1 = x0
            x0 = tmp

        x0 = x0.to(device)
        x1 = x1.to(device)

        if global_step > 0 and global_step % 300 == 0:
            # reset group features and weights every 300 iterations
            model._reset_group_features()
            model._reset_momentum_weights()
        else:
            # update momentum
            utils.update_momentum(model.backbone, model.backbone_momentum, 0.99)
            utils.update_momentum(
                model.projection_head, model.projection_head_momentum, 0.99
            )

        x0_features = model.backbone(x0).flatten(start_dim=1)
        x0_encoded = model.projection_head(x0_features)
        x0_predicted = model.prediction_head(x0_encoded)
        x1_features = model.backbone_momentum(x1).flatten(start_dim=1)
        x1_encoded = model.projection_head_momentum(x1_features)

        # update group features and get group assignments
        assignments = model.smog.assign_groups(x1_encoded)
        model.smog.update_groups(x0_encoded)

        logits = model.smog(x0_predicted, temperature=0.1)
        loss = criterion(logits, assignments)

        # use memory bank to periodically reset the group features with k-means
        memory_bank(x0_encoded, update=True)

        global_step += 1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
