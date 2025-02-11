from typing import Any, Tuple

from lightly.utils.benchmarking.linear_classifier import LinearClassifier


class FinetuneClassifier(LinearClassifier):
    def __init__(
        self,
        model: Any,
        batch_size_per_device: int,
        lr: float = 0.05,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
        freeze_model: bool = False,
    ) -> None:
        super(FinetuneClassifier, self).__init__(
            model,
            batch_size_per_device,
            lr,
            feature_dim,
            num_classes,
            topk,
            freeze_model,
        )
