from typing import Any
from torch.nn import Linear, MSELoss
from torch.optim import SGD
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from lightly.utils import dist
import torch
from pytorch_lightning.strategies.ddp import DDPStrategy
import time
import os





class Model(LightningModule):
    def __init__(self, gather: bool):
        super().__init__()
        self.model = Linear(5, 2, bias=False)
        # initialize the weights to be an arange
        self.model.weight.data = torch.arange(5*2).reshape(2,5).float()
        self.criterion = MSELoss()
        self.gather = gather

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x = batch[0]
        y = batch[1]
        self.print_("training_step")
        x = self.model(x)
        if self.gather:
            x = torch.cat(dist.gather(x))
            with torch.no_grad():
                y = torch.cat(dist.gather(y))
        
        loss = self.criterion(x, y)
        self.print_("training_step after gather", loss = loss.item(), x=x, y=y)
        return loss
    
    def on_after_backward(self) -> None:
        result = super().on_after_backward()
        self.print_("on_after_backward")
        return result
    
    def print_(self, step, **kwargs):
        p = list(self.parameters())[0]
        print(
            f"{step}, {self.local_rank=}, {self.global_rank=}, {p.grad=}, {p=}, {kwargs=}"
        )

    def configure_optimizers(self) -> Any:
        return SGD(self.parameters(), lr=1)

def test_gather() -> None:

    params_0_devices = None
    for n_devices in [1, 1, 2,2]:
        pl.seed_everything(42, workers=True)
        batch_size = 4 / n_devices
        assert int(batch_size) == batch_size
        batch_size = int(batch_size)

        xs = torch.arange(4 * 2 * 5).reshape(4, 2, 5).float()
        ys = torch.arange(4 * 2 * 2).reshape(4, 2, 2).float()
        if False:
            xs = torch.ones(4, 2, 5).float()
            ys = torch.ones(4, 2, 2).float()


        dataset = TensorDataset(xs, ys)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        model = Model(gather=n_devices > 1)

        trainer = Trainer(
            devices=n_devices,
            accelerator="cpu",
            strategy=DDPStrategy(find_unused_parameters=False),
            max_epochs=1,
        )
        trainer.fit(
            model,
            dataloader,
        )

        if os.getenv("LOCAL_RANK", '0') == '0':
            print()

            print(f"Finished training with {n_devices=}")
            print("Model parameters:")
            print(list(model.parameters()))
            if params_0_devices is None:
                params_0_devices = [param.detach() for param in model.parameters()]
            else:
                for p1, p2 in zip(params_0_devices, model.parameters()):
                    assert torch.allclose(p1, p2), f"{p1} != {p2}"

    print("Test passed")


if __name__ == "__main__":
    test_gather()