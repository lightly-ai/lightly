import torch

from lightly.utils.benchmarking import topk


def test_mean_topk_accuracy() -> None:
    predicted_classes = torch.tensor(
        [
            [1, 2, 3, 4],
            [4, 1, 10, 0],
            [3, 1, 5, 8],
        ]
    )
    targets = torch.tensor([1, 10, 8])
    assert topk.mean_topk_accuracy(predicted_classes, targets, k=(1, 2, 3, 4, 5)) == {
        1: 1 / 3,
        2: 1 / 3,
        3: 2 / 3,
        4: 1.0,
        5: 1.0,
    }
