import math

import pytest
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


@pytest.mark.parametrize(
    ("dtype", "knn_t"),
    [
        (torch.float16, 0.07),
        (torch.bfloat16, 0.01),
        (torch.float32, 0.01),
        (torch.float64, 0.001),
    ],
)
def test_knn__stable_weights_avoid_overflow(dtype: torch.dtype, knn_t: float) -> None:
    feature = torch.tensor([[1.0, 0.0]], dtype=dtype)
    feature_bank = torch.tensor(
        [[1.0, 0.9, 0.0], [0.0, math.sqrt(1.0 - 0.9**2), 1.0]], dtype=dtype
    )
    feature_labels = torch.tensor([1, 1, 0])

    pred_labels = knn.knn_predict(
        feature=feature,
        feature_bank=feature_bank,
        feature_labels=feature_labels,
        num_classes=2,
        knn_k=3,
        knn_t=knn_t,
    )

    assert pred_labels.tolist() == [[1, 0]]


def test_knn__stable_weights_are_centered_per_query() -> None:
    feature = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    feature_bank = torch.tensor([[1.0, 0.9, 0.0], [0.0, math.sqrt(1.0 - 0.9**2), 1.0]])
    feature_labels = torch.tensor([1, 0, 1])

    pred_labels = knn.knn_predict(
        feature=feature,
        feature_bank=feature_bank,
        feature_labels=feature_labels,
        num_classes=2,
        knn_k=3,
        knn_t=0.005,
    )

    assert pred_labels.tolist() == [[1, 0], [1, 0]]


def test_knn__half_precision_weights_preserve_class_ranking() -> None:
    feature = torch.tensor([[1.0, 0.0]], dtype=torch.float16)
    feature_bank = torch.tensor(
        [[1.0, 0.81, 0.8], [0.0, math.sqrt(1.0 - 0.81**2), 0.6]],
        dtype=torch.float16,
    )
    feature_labels = torch.tensor([2, 1, 0])

    pred_labels = knn.knn_predict(
        feature=feature,
        feature_bank=feature_bank,
        feature_labels=feature_labels,
        num_classes=3,
        knn_k=3,
        knn_t=0.01,
    )

    assert pred_labels.tolist() == [[2, 1, 0]]


def test_knn__stable_weights_preserve_all_negative_similarities() -> None:
    feature = torch.tensor([[-1.0]])
    feature_bank = torch.tensor([[0.2, 0.21, 0.22]])
    feature_labels = torch.tensor([1, 0, 0])

    pred_labels = knn.knn_predict(
        feature=feature,
        feature_bank=feature_bank,
        feature_labels=feature_labels,
        num_classes=2,
        knn_k=3,
        knn_t=0.001,
    )

    assert pred_labels.tolist() == [[1, 0]]
