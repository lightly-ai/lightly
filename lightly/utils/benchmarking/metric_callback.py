from collections import defaultdict
from typing import List

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch import Tensor


class MetricCallback(Callback):
    """Callback that collects log metrics from the LightningModule and stores them after
    every epoch.

    Example::

        >>> from lightly.utils.benchmarking import MetricCallback
        >>> from pytorch_lightning import LightningModule, Trainer
        >>>
        >>> class Model(LightningModule):
        >>>     def training_step(self, batch, batch_idx):
        >>>         ...
        >>>         self.log("train_acc", acc)
        >>>         ...
        >>>
        >>> metric_callback = MetricCallback()
        >>> trainer = Trainer(callbacks=[metric_callback], max_epochs=10)
        >>> trainer.fit(Model(), train_dataloder, val_dataloader)
        >>>
        >>> max_train_acc = max(metric_callback.get("train_acc"))
    """

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
            # Only store scalar values.
            if isinstance(value, float) or (
                isinstance(value, Tensor) and value.numel() == 1
            ):
                self.metrics[name].append(value)
