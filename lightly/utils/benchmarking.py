""" Helper modules for benchmarking SSL models """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl

# code for kNN prediction from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb


def knn_predict(feature: torch.Tensor,
                feature_bank: torch.Tensor,
                feature_labels: torch.Tensor, 
                num_classes: int,
                knn_k: int=200,
                knn_t: float=0.1) -> torch.Tensor:
    """Run kNN predictions on features based on a feature bank

    This method is commonly used to monitor performance of self-supervised
    learning methods.

    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.

    Args:
        feature: 
            Tensor of shape [N, D] for which you want predictions
        feature_bank: 
            Tensor of a database of features used for kNN
        feature_labels: 
            Labels for the features in our feature_bank
        num_classes: 
            Number of classes (e.g. `10` for CIFAR-10)
        knn_k: 
            Number of k neighbors used for kNN
        knn_t: 
            Temperature parameter to reweights similarities for kNN

    Returns:
        A tensor containing the kNN predictions

    Examples:
        >>> images, targets, _ = batch
        >>> feature = backbone(images).squeeze()
        >>> # we recommend to normalize the features
        >>> feature = F.normalize(feature, dim=1)
        >>> pred_labels = knn_predict(
        >>>     feature,
        >>>     feature_bank,
        >>>     targets_bank,
        >>>     num_classes=10,
        >>> )
    """

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(
        feature.size(0), -1), dim=-1, index=sim_indices)
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(
        0) * knn_k, num_classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(
        0), -1, num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


class BenchmarkModule(pl.LightningModule):
    """A PyTorch Lightning Module for automated kNN callback

    At the end of every training epoch we create a feature bank by feeding the
    `dataloader_kNN` passed to the module through the backbone.
    At every validation step we predict features on the validation data.
    After all predictions on validation data (validation_epoch_end) we evaluate
    the predictions on a kNN classifier on the validation data using the
    feature_bank features from the train data.

    We can access the highest test accuracy during a kNN prediction 
    using the `max_accuracy` attribute.

    Attributes:
        backbone:
            The backbone model used for kNN validation. Make sure that you set the
            backbone when inheriting from `BenchmarkModule`.
        max_accuracy:
            Floating point number between 0.0 and 1.0 representing the maximum
            test accuracy the benchmarked model has achieved.
        dataloader_kNN:
            Dataloader to be used after each training epoch to create feature bank.
        num_classes:
            Number of classes. E.g. for cifar10 we have 10 classes. (default: 10)
        knn_k:
            Number of nearest neighbors for kNN
        knn_t:
            Temperature parameter for kNN

    Examples:
        >>> class SimSiamModel(BenchmarkingModule):
        >>>     def __init__(dataloader_kNN, num_classes):
        >>>         super().__init__(dataloader_kNN, num_classes)
        >>>         resnet = lightly.models.ResNetGenerator('resnet-18')
        >>>         self.backbone = nn.Sequential(
        >>>             *list(resnet.children())[:-1],
        >>>             nn.AdaptiveAvgPool2d(1),
        >>>         )
        >>>         self.resnet_simsiam = 
        >>>             lightly.models.SimSiam(self.backbone, num_ftrs=512)
        >>>         self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
        >>>
        >>>     def forward(self, x):
        >>>         self.resnet_simsiam(x)
        >>>
        >>>     def training_step(self, batch, batch_idx):
        >>>         (x0, x1), _, _ = batch
        >>>         x0, x1 = self.resnet_simsiam(x0, x1)
        >>>         loss = self.criterion(x0, x1)
        >>>         return loss
        >>>     def configure_optimizers(self):
        >>>         optim = torch.optim.SGD(
        >>>             self.resnet_simsiam.parameters(), lr=6e-2, momentum=0.9
        >>>         )
        >>>         return [optim]
        >>>
        >>> model = SimSiamModel(dataloader_train_kNN)
        >>> trainer = pl.Trainer()
        >>> trainer.fit(
        >>>     model,
        >>>     train_dataloader=dataloader_train_ssl,
        >>>     val_dataloaders=dataloader_test
        >>> )
        >>> # you can get the peak accuracy using
        >>> print(model.max_accuracy)

    """

    def __init__(self,
                 dataloader_kNN: DataLoader,
                 num_classes: int,
                 knn_k: int=200,
                 knn_t: float=0.1):
        super().__init__()
        self.backbone = nn.Module()
        self.max_accuracy = 0.0
        self.dataloader_kNN = dataloader_kNN
        #self.gpus = gpus
        self.num_classes = num_classes
        self.knn_k = knn_k
        self.knn_t = knn_t

        # create dummy param to keep track of the device the model is using
        self.dummy_param = nn.Parameter(torch.empty(0))

    def training_epoch_end(self, outputs):
        # update feature bank at the end of each training epoch
        self.backbone.eval()
        self.feature_bank = []
        self.targets_bank = []
        with torch.no_grad():
            for data in self.dataloader_kNN:
                img, target, _ = data
                img = img.to(self.dummy_param.device)
                target = target.to(self.dummy_param.device)
                feature = self.backbone(img).squeeze()
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(target)
        self.feature_bank = torch.cat(
            self.feature_bank, dim=0).t().contiguous()
        self.targets_bank = torch.cat(
            self.targets_bank, dim=0).t().contiguous()
        self.backbone.train()

    def validation_step(self, batch, batch_idx):
        # we can only do kNN predictions once we have a feature bank
        if hasattr(self, 'feature_bank') and hasattr(self, 'targets_bank'):
            images, targets, _ = batch
            feature = self.backbone(images).squeeze()
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(
                feature,
                self.feature_bank,
                self.targets_bank,
                self.num_classes,
                self.knn_k,
                self.knn_t
            )
            num = images.size(0)
            top1 = (pred_labels[:, 0] == targets).float().sum().item()
            return (num, top1)

    def validation_epoch_end(self, outputs):
        if outputs:
            total_num = 0
            total_top1 = 0.
            for (num, top1) in outputs:
                total_num += num
                total_top1 += top1
            acc = float(total_top1 / total_num)
            if acc > self.max_accuracy:
                self.max_accuracy = acc
            self.log('kNN_accuracy', acc * 100.0, prog_bar=True)
