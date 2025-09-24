from typing import Dict, List, Union

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch import Tensor

MetricValue = Union[Tensor, float]


class MetricCallback(Callback):
    """Callback that collects log metrics from the LightningModule and stores them after
    every epoch.

    Attributes:
        train_metrics:
            Dictionary that stores the last logged metrics after every train epoch.
        val_metrics:
            Dictionary that stores the last logged metrics after every validation epoch.

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
        >>>     def validation_step(self, batch, batch_idx):
        >>>         ...
        >>>         self.log("val_acc", acc)
        >>>         ...
        >>>
        >>> metric_callback = MetricCallback()
        >>> trainer = Trainer(callbacks=[metric_callback], max_epochs=10)
        >>> trainer.fit(Model(), train_dataloder, val_dataloader)
        >>>
        >>> max_train_acc = max(metric_callback.train_metrics["train_acc"])
        >>> max_val_acc = max(metric_callback.val_metrics["val_acc"])
    """

    def __init__(self) -> None:
        super().__init__()
        self.train_metrics: Dict[str, List[float]] = {}
        self.val_metrics: Dict[str, List[float]] = {}

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.sanity_checking:
            self._append_metrics(metrics_dict=self.train_metrics, trainer=trainer)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not trainer.sanity_checking:
            self._append_metrics(metrics_dict=self.val_metrics, trainer=trainer)

    def _append_metrics(
        self, metrics_dict: Dict[str, List[float]], trainer: Trainer
    ) -> None:
        for name, value in trainer.callback_metrics.items():
            if isinstance(value, Tensor) and value.numel() != 1:
                # Skip non-scalar tensors.
                continue
            metrics_dict.setdefault(name, []).append(float(value))
