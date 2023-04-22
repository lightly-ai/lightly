from typing import Dict, Sequence

import torch
from torch import Tensor


def mean_topk_accuracy(
    predicted_classes: Tensor, targets: Tensor, k: Sequence[int]
) -> Dict[int, Tensor]:
    """Computes the mean accuracy for the specified values of k."""
    accuracy = {}
    targets = targets.unsqueeze(1)
    with torch.no_grad():
        for num_k in k:
            correct = torch.eq(predicted_classes[:, :num_k], targets)
            accuracy[num_k] = correct.float().sum() / targets.shape[0]
    return accuracy
