from omegaconf import DictConfig
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from torchvision import models
from torch.nn.modules import Identity
from lightly.pretrain.callbacks.onnx_checkpoint import ONNXCheckpoint
from lightly.pretrain.methods import methods
import torchvision.transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import Module, Linear


def pretrain_from_cfg(cfg: DictConfig) -> None:
    # Create and prepare backbone.
    backbone = models.get_model(name=cfg.backbone)
    feature_dim = _get_feature_dim(backbone)
    _disable_classifier(backbone)

    # Create ssl model.
    model = methods.get_model(name=cfg.method)(
        backbone=backbone,
        feature_dim=feature_dim,
        batch_size_per_device=cfg.loader.train.batch_size,
        num_classes=cfg.dataset.num_classes,
    )

    # Create train dataset.
    train_transform_fn = methods.get_transform(name=cfg.method)
    if cfg.transform.train:
        train_transform = train_transform_fn(**cfg.transform.train)
    else:
        train_transform = train_transform_fn()
    train_dataset = ImageFolder(root=cfg.dataset.train_dir, transform=train_transform)

    # Create val dataset.
    val_transform = T.Compose(
        [
            T.Resize(cfg.transform.val.resize.size),
            T.CenterCrop(cfg.transform.val.center_crop.size),
            T.ToTensor(),
            T.Normalize(
                mean=cfg.transform.val.normalize.mean,
                std=cfg.transform.val.normalize.std,
            ),
        ]
    )
    val_dataset = (
        None
        if cfg.dataset.val_dir is None
        else ImageFolder(root=cfg.dataset.val_dir, transform=val_transform)
    )

    # Create dataloaders.
    train_dataloader = DataLoader(dataset=train_dataset, **cfg.loader.train)
    val_dataloader = (
        None
        if val_dataset is None
        else DataLoader(dataset=val_dataset, **cfg.loader.val)
    )

    # Create trainer.
    trainer_kwargs = dict(cfg.trainer)
    trainer_kwargs.pop("callbacks", None)
    trainer_kwargs.pop("logger", None)
    trainer = Trainer(
        **trainer_kwargs,
        logger=[TensorBoardLogger(**cfg.trainer.logger.tensorboard)],
        callbacks=[
            ModelCheckpoint(**cfg.trainer.callbacks.model_checkpoint),
            ONNXCheckpoint(
                image_size=cfg.transform.val.resize.size,
                **cfg.trainer.callbacks.onnx_checkpoint
            ),
        ]
    )

    # Train.
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def _get_feature_dim(backbone: Module) -> int:
    classifier = _get_classifier(backbone)
    linear = _find_linear(classifier)
    return int(linear.weight.shape[1])


def _get_classifier(mod: Module) -> Module:
    try:
        return mod.classifier
    except AttributeError:
        return mod.fc


def _disable_classifier(mod: Module):
    try:
        _ = mod.classifier
        mod.classifier = Identity()
    except AttributeError:
        mod.fc = Identity()


def _find_linear(mod: Module) -> Module:
    modules = [mod] + list(mod.modules())
    for m in modules:
        if isinstance(m, Linear):
            return m
    raise ValueError("No linear layer found in model.")
