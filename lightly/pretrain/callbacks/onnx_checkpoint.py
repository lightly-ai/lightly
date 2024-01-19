from pytorch_lightning.callbacks import Checkpoint
from torch import onnx
from pytorch_lightning import Trainer, LightningModule
from pathlib import Path
import torch


class ONNXCheckpoint(Checkpoint):
    def __init__(
        self, dirpath: str, filename: str, every_n_epochs: int, image_size: int
    ):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.every_n_epochs = every_n_epochs
        self.image_size = image_size
        self._current_epoch = 0

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save a checkpoint at the end of the training epoch."""
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self._save_onnx(trainer, pl_module)

    def _save_onnx(self, trainer: Trainer, pl_module: LightningModule) -> None:
        backbone = pl_module.backbone
        training_mode = backbone.training
        # Set to eval for onnx export.
        backbone.eval()
        x = torch.randn(
            1,
            3,
            self.image_size,
            self.image_size,
            requires_grad=True,
            device=pl_module.device,
        )
        filepath = Path(self.dirpath) / self.filename.format(
            epoch=trainer.current_epoch
        )
        onnx.export(
            model=backbone,
            args=x,
            f=str(filepath),
            input_names=["input"],
            output_names=["output"],
        )
        # Reset to previous training state.
        backbone.train(mode=training_mode)
