"""One row per dataset SimCLR is benchmarked on.

Every row states every field. No defaults, no inheritance, no merging, and
nothing computed from anything else: two rows read side by side show every
difference there is, and a field left out is a type error rather than a silent
fallback. That is what keeps this a table rather than a configuration system,
and it is why ``lr`` is a literal instead of a scaling rule.

Nothing here ships. ``benchmarks/`` is not in the wheel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from lightly.transforms.utils import CIFAR10_NORMALIZE, IMAGENET_NORMALIZE


@dataclass(frozen=True)
class Setting:
    """One dataset SimCLR is benchmarked on.

    Attributes:
        name: The row's key in ``SETTINGS``.
        size: Crop size in pixels.
        normalize: Channel means and standard deviations.
        stem: ``standard`` or ``small_image``, the 3x3 stride-1 ResNet stem.
        blur: Probability of Gaussian blur.
        color_jitter_strength: Multiplies the four colour-jitter strengths.
        backbone: ``resnet18`` or ``resnet50``.
        feature_dim: The backbone's output width, and the head's input.
        hidden_dim: The projection head's hidden width.
        out_dim: The projection head's output width.
        batch_size: Total batch size, summed over ranks.
        epochs: Pretraining epochs.
        temperature: NT-Xent temperature.
        weight_decay: Applied to everything except norms and biases.
        optimizer: ``lars`` or ``sgd``.
        lr: The learning rate at this row's ``batch_size``. A literal, not a rule.
        momentum: Optimiser momentum.
        warmup_epochs: Linear warmup before the cosine decay.
        num_classes: Classes in the dataset, for the probes.
        knn_k: Neighbours the kNN probe votes over.
        knn_t: Temperature the kNN probe reweights similarities with.
    """

    name: str
    size: int
    normalize: Dict[str, List[float]]
    stem: str
    blur: float
    color_jitter_strength: float
    backbone: str
    feature_dim: int
    hidden_dim: int
    out_dim: int
    batch_size: int
    epochs: int
    temperature: float
    weight_decay: float
    optimizer: str
    lr: float
    momentum: float
    warmup_epochs: int
    num_classes: int
    knn_k: int
    knn_t: float


# Chen et al. 2020, table 6 and appendix B.1. lr is the paper's linear rule at
# this batch size: 0.3 * 4096 / 256.
IMAGENET = Setting(
    name="imagenet",
    size=224,
    normalize=IMAGENET_NORMALIZE,
    stem="standard",
    blur=0.5,
    color_jitter_strength=1.0,
    backbone="resnet50",
    feature_dim=2048,
    hidden_dim=2048,
    out_dim=128,
    batch_size=4096,
    epochs=100,
    temperature=0.1,
    weight_decay=1e-6,
    optimizer="lars",
    lr=4.8,
    momentum=0.9,
    warmup_epochs=10,
    num_classes=1000,
    knn_k=200,
    knn_t=0.1,
)

# The settings examples/simclr.py is written out with. Small enough for one GPU.
CIFAR10 = Setting(
    name="cifar10",
    size=32,
    normalize=CIFAR10_NORMALIZE,
    stem="small_image",
    blur=0.0,
    color_jitter_strength=0.5,
    backbone="resnet18",
    feature_dim=512,
    hidden_dim=512,
    out_dim=128,
    batch_size=256,
    epochs=100,
    temperature=0.5,
    weight_decay=5e-4,
    optimizer="sgd",
    lr=0.06,
    momentum=0.9,
    warmup_epochs=0,
    num_classes=10,
    knn_k=200,
    knn_t=0.1,
)

SETTINGS = {setting.name: setting for setting in (IMAGENET, CIFAR10)}
