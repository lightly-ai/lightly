from typing import Sequence

import torch
from torch import Tensor


def topk_accuracy(
    predictions: Tensor, targets: Tensor, k: Sequence[int]
) -> list[Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k."""
    accuracy = []
    targets = targets.unsqueeze(1)
    with torch.no_grad():
        maxk = max(k)
        _, topk_labels = predictions.topk(maxk, -1, True, True)
        for num_k in k:
            correct = torch.eq(topk_labels[:, :num_k], targets)
            acc = correct.float().mean(dim=-1)
            accuracy.append(acc)
    return accuracy
