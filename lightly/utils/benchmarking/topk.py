from typing import Dict, Sequence

import torch
from torch import Tensor


def mean_topk_accuracy(
    predicted_classes: Tensor, targets: Tensor, k: Sequence[int]
) -> Dict[int, Tensor]:
    """Computes the mean accuracy for the specified values of k.

    The mean is calculated over the batch dimension.

    Args:
        predicted_classes:
            Tensor of shape (batch_size, num_classes) with the predicted classes sorted
            in descending order of confidence.
        targets:
            Tensor of shape (batch_size) containing the target classes.
        k:
            Sequence of integers specifying the values of k for which the accuracy
            should be computed.

    Returns:
        Dictionary containing the mean accuracy for each value of k. For example for
        k=(1, 5) the dictionary could look like this: {1: 0.4, 5: 0.6}.
    """
    accuracy = {}
    targets = targets.unsqueeze(1)
    with torch.no_grad():
        for num_k in k:
            correct = torch.eq(predicted_classes[:, :num_k], targets)
            accuracy[num_k] = correct.float().sum() / targets.shape[0]
    return accuracy
