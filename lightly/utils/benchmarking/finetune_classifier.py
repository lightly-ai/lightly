from torch.optim import SGD

from lightly.utils.benchmarking import LinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler


class FinetuneClassifier(LinearClassifier):
    def configure_optimizers(self):
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
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
