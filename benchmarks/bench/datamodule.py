"""The data side of a benchmark: loaders, and nothing that touches the update."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightly.data import LightlyDataset
from lightly.data.sample import collate
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T


class ImageDataModule(LightningDataModule):
    """Train, kNN-train and validation loaders over two image folders.

    The kNN loader reads the training images under the validation transform, so
    the probe scores the encoder rather than the augmentations. It is the first
    validation loader, and the validation set is the second; the probes read that
    ordering.

    Args:
        train_dir: Folder of training images.
        val_dir: Folder of validation images.
        transform: The method's transform. It returns views.
        batch_size: Per-device batch size.
        size: Crop size the validation transform resizes to.
        normalize: Channel means and standard deviations.
        num_workers: Worker processes per loader.
    """

    def __init__(
        self,
        train_dir: Path,
        val_dir: Path,
        transform: Callable[..., Any],
        batch_size: int,
        size: int,
        normalize: Any,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_transform = T.Compose(
            [
                T.Resize(int(size * 256 / 224)),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

    def _loader(
        self,
        directory: Path,
        transform: Callable[..., Any],
        shuffle: bool,
        drop_last: bool,
        collate_fn: Optional[Callable[..., Any]] = None,
    ) -> DataLoader:
        return DataLoader(
            LightlyDataset(input_dir=str(directory), transform=transform),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(
            self.train_dir,
            self.transform,
            shuffle=True,
            drop_last=True,
            collate_fn=collate,
        )

    def val_dataloader(self) -> List[DataLoader]:
        knn_train = self._loader(
            self.train_dir, self.val_transform, shuffle=False, drop_last=False
        )
        validation = self._loader(
            self.val_dir, self.val_transform, shuffle=False, drop_last=False
        )
        return [knn_train, validation]
