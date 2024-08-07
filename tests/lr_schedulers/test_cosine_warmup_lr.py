import pytest
from torch.nn import Linear
from torch.optim import SGD

from lightly.lr_schedulers import CosineWarmupLR
from lightly.schedulers import CosineWarmupScheduler


class TestCosineWarmupLR:
    def test__equivalence(self) -> None:
        """Test that CosineWarmupLR and CosineWarmupScheduler are equivalent."""
        model = Linear(10, 1)
        optimizer = SGD(model.parameters(), lr=1.0, momentum=0.0, weight_decay=0.0)
        lr_scheduler = CosineWarmupLR(
            optimizer, warmup_epochs=3, max_epochs=6, verbose=True, end_value=0.0
        )
        scheduler = CosineWarmupScheduler(max_steps=6, end_value=0.0, warmup_steps=3)

        for step in range(6):
            # Use last_step because LR scheduler uses previous epoch to calculate lr
            # at the current step.
            assert lr_scheduler.get_lr()[0] == scheduler.get_value(
                step=scheduler.last_step
            )
            assert lr_scheduler.scale_lr(epoch=step) == scheduler.get_value(step=step)
            optimizer.step()
            lr_scheduler.step()
            scheduler.step()

        # step > max_epochs
        with pytest.warns(
            RuntimeWarning, match="Current step number 7 exceeds max_steps 6."
        ):
            lr_scheduler.step()
