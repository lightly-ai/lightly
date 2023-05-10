import torch
import torch.nn.functional as F

from lightly.utils.benchmarking import knn


def test_knn() -> None:
    feature_bank = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ]
    ).t()
    feature_labels = torch.tensor([0, 1, 2, 3])
    features = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [-1.0, 1.1, -1.0],
        ]
    )
    feature_bank = F.normalize(feature_bank, dim=0)
    features = F.normalize(features, dim=1)
    pred_labels = knn.knn_predict(
        feature=features,
        feature_bank=feature_bank,
        feature_labels=feature_labels,
        num_classes=4,
        knn_k=4,
    )
    assert pred_labels.tolist() == [
        [0, 1, 2, 3],
        [1, 3, 0, 2],
    ]


def test_knn__knn_k() -> None:
    feature_bank = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ]
    ).t()
    feature_labels = torch.tensor([0, 1, 0, 1])
    features = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [-1.0, 1.1, -1.0],
        ]
    )
    feature_bank = F.normalize(feature_bank, dim=0)
    features = F.normalize(features, dim=1)
    pred_labels = knn.knn_predict(
        feature=features,
        feature_bank=feature_bank,
        feature_labels=feature_labels,
        num_classes=4,
        knn_k=2,
    )
    assert pred_labels.tolist() == [
        [0, 1, 2, 3],
        # 1 is first because bank features with index 1 and 3 and label 1 are closest.
        # 0 is second because bank features with index 0 and 2 and label 0 are 2nd closest.
        # 2 and 3 are last because there are no bank features with label 2 or 3.
        [1, 0, 2, 3],
    ]
