"""Probes a benchmark scores its encoder with.

These take features and targets and return numbers. They do not inherit from
``LightningModule``, do not log, and do not key their lifecycle on which
dataloader a batch came from: the benchmark says when the bank is built and when
it is scored, on two adjacent lines a reviewer can read.

The public, method-agnostic version of this is ``lightly.eval``, which is a
separate piece of work. Until it lands this is the smallest thing that serves the
benchmarks in this tree.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import normalize

from lightly.utils.benchmarking.knn import knn_predict
from lightly.utils.benchmarking.topk import mean_topk_accuracy

__all__ = ["KNNProbe"]


class KNNProbe:
    """Weighted kNN over a feature bank built from the training set.

    Settings follow InstDisc, which is what the published benchmark numbers used.

    Args:
        num_classes: Classes in the dataset.
        k: Neighbours that vote.
        t: Temperature the similarities are reweighted with.
        topk: Which top-k accuracies to report.
        feature_dtype: Bank dtype. ``float16`` halves the memory.
        normalize_features: Whether to L2-normalize before the search.
    """

    def __init__(
        self,
        num_classes: int,
        k: int,
        t: float,
        topk: Tuple[int, ...] = (1, 5),
        feature_dtype: torch.dtype = torch.float32,
        normalize_features: bool = True,
    ) -> None:
        self.num_classes = num_classes
        self.k = k
        self.t = t
        self.topk = topk
        self.feature_dtype = feature_dtype
        self.normalize_features = normalize_features
        self._features: List[Tensor] = []
        self._targets: List[Tensor] = []
        self._bank: Optional[Tensor] = None
        self._labels: Optional[Tensor] = None

    def _prepare(self, features: Tensor) -> Tensor:
        if self.normalize_features:
            features = normalize(features, dim=1)
        return features.to(self.feature_dtype)

    def reset(self) -> None:
        """Drops the bank, so the next epoch starts from nothing."""
        self._features = []
        self._targets = []
        self._bank = None
        self._labels = None

    def add(self, features: Tensor, targets: Tensor) -> None:
        """Adds one batch of training features to the bank.

        Args:
            features: Encoder features of shape ``(B, D)``.
            targets: Labels of shape ``(B,)``.
        """
        self._features.append(self._prepare(features).cpu())
        self._targets.append(targets.cpu())

    def build(
        self,
        gather: Optional[Callable[[Tensor], Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Closes the bank so it can be scored against.

        Args:
            gather:
                Collects a tensor from every rank, returning
                ``(world_size, B, ...)``. ``LightningModule.all_gather`` is one.
                Omit it on a single rank.
            device: Where the bank is scored. Defaults to where it was built.

        Raises:
            ValueError: If no features were added.
        """
        if not self._features:
            raise ValueError("the kNN bank is empty: call add() before build()")
        features = torch.cat(self._features, dim=0)
        targets = torch.cat(self._targets, dim=0)
        if gather is not None:
            features = gather(features)
            targets = gather(targets)
        # (dim, world_size * batch) is the layout knn_predict reads.
        self._bank = features.flatten(end_dim=-2).t().contiguous().to(device)
        self._labels = targets.flatten().contiguous().to(device)
        self._features = []
        self._targets = []

    def score(self, features: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        """Scores one batch against the bank.

        Args:
            features: Encoder features of shape ``(B, D)``.
            targets: Labels of shape ``(B,)``.

        Returns:
            One entry per top-k, keyed ``val_knn_top{k}``.

        Raises:
            ValueError: If the bank has not been built.
        """
        if self._bank is None or self._labels is None:
            raise ValueError("the kNN bank is not built: call build() before score()")
        predicted = knn_predict(
            feature=self._prepare(features),
            feature_bank=self._bank,
            feature_labels=self._labels,
            num_classes=self.num_classes,
            knn_k=self.k,
            knn_t=self.t,
        )
        topk = mean_topk_accuracy(
            predicted_classes=predicted, targets=targets, k=self.topk
        )
        return {f"val_knn_top{k}": accuracy for k, accuracy in topk.items()}
