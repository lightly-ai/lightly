from typing import Any, Dict, List, Tuple, Union

from torch.optim import SGD, Optimizer

from lightly.utils.benchmarking import LinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler


class FinetuneClassifier(LinearClassifier):
    # Type ignore is needed because return type of LightningModule.configure_optimizers
    # is complicated and typing changes between versions.
    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> Tuple[List[Optimizer], List[Dict[str, Union[Any, str]]]]:
        parameters = list(self.classification_head.parameters())
        parameters += self.model.parameters()
        optimizer = SGD(
            parameters,
            lr=0.05 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
