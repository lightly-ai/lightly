from collections import defaultdict

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from typing import List


class MetricCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = defaultdict(list)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._append_metrics(trainer=trainer)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._append_metrics(trainer=trainer)

    def get(self, name: str) -> List[float]:
        return [float(v) for v in self.metrics[name]]

    def _append_metrics(self, trainer: Trainer) -> None:
        for name, value in trainer.callback_metrics.items():
            if isinstance(value, float) or (
                isinstance(value, Tensor) and value.numel() == 1
            ):
                self.metrics[name].append(value)
