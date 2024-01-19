from lightly.transforms import SimCLRTransform, VICRegTransform
from lightly.pretrain.methods.simclr import SimCLR
from lightly.pretrain.methods.vicreg import VICReg
from typing import Type, Union


def get_model(name: str) -> Type[Union[SimCLR, VICReg]]:
    return {"simclr": SimCLR, "vicreg": VICReg}[name]


def get_transform(name: str) -> Type[Union[SimCLRTransform, VICRegTransform]]:
    return {"simclr": SimCLRTransform, "vicreg": VICRegTransform}[name]
