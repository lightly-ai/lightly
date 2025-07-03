# 为多标签分类任务重新设计的Simclr
# 屏蔽了训练阶段分类器部分的代码，保证其纯自监督训练过程

import math
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torchvision.models import resnet50

from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.transforms import SimCLRTransform
from lightly.utils.benchmarking import OnlineLinearClassifier_muti
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler, DelayedCosineAnnealingLR


class SimCLR(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device
        self.mode = 'eval'

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = SimCLRProjectionHead()
        self.criterion = NTXentLoss(temperature=0.1, gather_distributed=True)
        # self.criterion1 = SupConLoss(temperature=0.1)

        self.online_classifier = OnlineLinearClassifier_muti(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        # if self.mode == "eval":
        #     return torch.tensor(0.0, device='cuda', requires_grad=False)     # 跳过训练，直接返回0损失

        views, targets = batch[0], batch[1]
        features = self.forward(torch.cat(views)).flatten(start_dim=1)    # 2*batch,2048
        z = self.projection_head(features)     # 2b, 128
        z0, z1 = z.chunk(len(views))        # 拆分 8, 128
        loss = self.criterion(z0, z1)
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )
        # feature1 = self.forward(views).flatten(start_dim=1).detach()    # b,2048
        # # 用以评估性能的线性探测分类器的损失计算，实质上进行了监督学习，使用了真实标签，如果训练集中无标签，则此处targets全为零
        # cls_loss, cls_log = self.online_classifier.training_step(
        #     (feature1.detach(), targets.repeat_interleave(2, dim=0)), batch_idx
        # )
        # self.log(
        #     "train_loss", cls_loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        # )
        # self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        # self.manual_backward(loss)  # 反向传播，计算梯度
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f"grad/{name}", param.grad.norm(), prog_bar=False, sync_dist=True)
        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1).detach()  

        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        optimizer = LARS(
            [
                {"name": "simclr", "params": params},
                {
                    "name": "simclr_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Square root learning rate scaling improves performance for small
            # batch sizes (<=2048) and few training epochs (<=200). Alternatively,
            # linear scaling can be used for larger batches and longer training:
            #   lr=0.3 * self.batch_size_per_device * self.trainer.world_size / 256
            # See Appendix B.1. in the SimCLR paper https://arxiv.org/abs/2002.05709
            lr=0.075 * math.sqrt(self.batch_size_per_device * self.trainer.world_size),     # 每个设备的批次大小和分布式训练的总设备数
            # lr = 1.6e-3,
            momentum=0.9,
            # Note: Paper uses weight decay of 1e-6 but reference code 1e-4. See:
            # https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/README.md?plain=1#L103
            weight_decay=1e-4,
        )
        # 预热与余弦退火的学习策略，前10%学习率线性增加，后90%学习率余弦退火
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                # warmup_epochs=int(
                #     self.trainer.estimated_stepping_batches
                #     / self.trainer.max_epochs
                #     * 10
                # ),
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
    #     scheduler = {
    #     "scheduler": DelayedCosineAnnealingLR(
    #         optimizer=optimizer,
    #         start_epoch=10,  # 从第10个epoch开始衰减
    #         end_epoch=100,   # 到第100个epoch结束衰减
    #     ),
    #     "interval": "epoch",  # 按epoch更新
    # }
        return [optimizer], [scheduler]


transform = SimCLRTransform()
