from typing import Dict, Tuple

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Sigmoid, BCEWithLogitsLoss

from lightly.utils.benchmarking.topk import mean_topk_accuracy

import torch
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, average_precision_score
from itertools import product

class OnlineLinearClassifier_muti(LightningModule):
    def __init__(
        self,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.topk = topk
        self.s = Sigmoid()
        self.classification_head = Linear(feature_dim, num_classes)
        self.criterion = BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.classification_head(x.detach().flatten(start_dim=1))

    def shared_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[int, Tensor]]:
        features, targets = batch[0], batch[1]
        predictions = self.forward(features)
        if targets.dtype != torch.float16 or targets.dtype != torch.float16:
            targets = targets.half()  # 半浮点精度
        if targets.shape[0] == 2 * features.shape[0]:
            targets = targets.chunk(2, dim=0)[0]
        loss = self.criterion(predictions, targets)     # 更改BCEWithLogitsLoss()
        predictions_sigmoid = self.s(predictions)
        metric_dict = self.calculate_multilabel_metrics(predictions_sigmoid, targets)
        # _, predicted_classes = predictions.topk(max(self.topk))
        # topk = mean_topk_accuracy(predicted_classes, targets, k=self.topk)
        return loss, metric_dict

    def training_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        log_dict = {"train_online_cls_loss": loss}
        log_dict.update({f"{k}": acc for k, acc in topk.items()})
        return loss, log_dict

    def validation_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        log_dict = {"val_online_cls_loss": loss}
        log_dict.update({f"eval/{k}": acc for k, acc in topk.items()})
        # log_dict.update({f"val_online_cls_top{k}": acc for k, acc in topk.items()})
        return loss, log_dict

    def calculate_multilabel_metrics(self, predictions, targets):
        """
            返回包含6种指标的字典
        """
        # 确保输入是numpy数组
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()

        batch_size, num_classes = predictions.shape

        # 1. 宏平均F1分数 (Macro-F1 Score)
        # 将概率预测转换为二进制标签 (使用0.5作为阈值)
        pred_binary = (predictions > 0.5).astype(int)

        # 计算每个类别的F1分数，然后取平均
        macro_f1 = f1_score(targets, pred_binary, average='macro')

        # 2. 子集准确率 (Subset Accuracy)
        # 检查预测标签集合与真实标签集合是否完全一致
        exact_matches = 0
        for pred, tgt in zip(pred_binary, targets):
            if np.array_equal(pred, tgt):
                exact_matches += 1
        subset_accuracy = exact_matches / batch_size

        # 3. 汉明损失 (Hamming Loss)
        hamming_loss_val = hamming_loss(targets, pred_binary)

        # 4. 标签级准确率 (Label-based Accuracy)
        # 汉明准确率 = 1 - 汉明损失
        label_accuracy = 1 - hamming_loss_val

        # 5. 平均精度均值 (mAP)
        mAP = 0.0
        for class_idx in range(num_classes):
            # 确保该类至少有一个正样本
            if np.sum(targets[:, class_idx]) > 0:
                # 计算该类的平均精度
                ap = average_precision_score(
                    targets[:, class_idx],
                    predictions[:, class_idx],
                    pos_label=1
                )
                mAP += ap
            else:
                # 如果该类没有正样本，AP设为0
                pass

        mAP = mAP / num_classes if num_classes > 0 else 0

        # 6. Jaccard相似系数 (Jaccard Similarity Coefficient)
        jaccard_scores = []
        for pred, tgt in zip(pred_binary, targets):
            # 计算交集
            intersection = np.sum(np.logical_and(pred, tgt))
            # 计算并集
            union = np.sum(np.logical_or(pred, tgt))
            # 避免除零错误
            if union == 0:
                jaccard_scores.append(1.0 if intersection == 0 else 0.0)
            else:
                jaccard_scores.append(intersection / union)

        avg_jaccard = np.mean(jaccard_scores)

        # 整理结果
        metrics = {
            'f1-macro': macro_f1,
            'subset_accuracy': subset_accuracy,
            'hamming_loss': hamming_loss_val,
            'acc': label_accuracy,
            'mAP': mAP,
            'jaccard': avg_jaccard
        }

        return metrics