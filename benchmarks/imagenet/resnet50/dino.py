import copy
from typing import List, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torchvision.models import resnet50

from lightly.loss import DINOLoss
from lightly.lr_schedulers import CosineWarmupLR
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import (
    activate_requires_grad,
    deactivate_requires_grad,
    get_weight_decay_parameters,
    update_momentum,
)
from lightly.schedulers import cosine_schedule
from lightly.transforms import DINOTransform
from lightly.utils.benchmarking import OnlineLinearClassifier


class DINO(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = DINOProjectionHead()
        self.student_backbone = copy.deepcopy(self.backbone)
        self.student_projection_head = DINOProjectionHead(freeze_last_layer=1)
        self.criterion = DINOLoss()

        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: Tensor) -> Tensor:
        features = self.student_backbone(x).flatten(start_dim=1)
        projections = self.student_projection_head(features)
        return projections

    def on_train_start(self) -> None:
        deactivate_requires_grad(self.backbone)
        deactivate_requires_grad(self.projection_head)

    def on_train_end(self) -> None:
        activate_requires_grad(self.backbone)
        activate_requires_grad(self.projection_head)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        # Momentum update teacher.
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.996,
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.backbone, m=momentum)
        update_momentum(self.student_projection_head, self.projection_head, m=momentum)

        views, targets = batch[0], batch[1]
        global_views = torch.cat(views[:2])
        local_views = torch.cat(views[2:])

        teacher_features = self.forward(global_views).flatten(start_dim=1)
        teacher_projections = self.projection_head(teacher_features)
        student_projections = torch.cat(
            [self.forward_student(global_views), self.forward_student(local_views)]
        )

        loss = self.criterion(
            teacher_out=teacher_projections.chunk(2),
            student_out=student_projections.chunk(len(views)),
            epoch=self.current_epoch,
        )
        self.log_dict(
            {"train_loss": loss, "ema_momentum": momentum},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(targets),
        )

        # Online classification.
        cls_loss, cls_log = self.online_classifier.training_step(
            (teacher_features.chunk(2)[0].detach(), targets), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.student_backbone, self.student_projection_head]
        )
        # For ResNet50 we use SGD instead of AdamW/LARS as recommended by the authors:
        # https://github.com/facebookresearch/dino#resnet-50-and-other-convnets-trainings
        optimizer = SGD(
            [
                {"name": "dino", "params": params},
                {
                    "name": "dino_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.03 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = {
            "scheduler": CosineWarmupLR(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Union[int, float, None] = None,
        gradient_clip_algorithm: Union[str, None] = None,
    ) -> None:
        self.clip_gradients(
            optimizer=optimizer,
            gradient_clip_val=3.0,
            gradient_clip_algorithm="norm",
        )
        self.student_projection_head.cancel_last_layer_gradients(self.current_epoch)

    def on_before_optimizer_step(self, optmizer) -> None:
        for group in optmizer.param_groups:
            wd_scheduler = group.get("weight_decay_scheduler")
            if wd_scheduler is not None:
                group["weight_decay"] = wd_scheduler(self.current_epoch)


# For ResNet50 we adjust crop scales as recommended by the authors:
# https://github.com/facebookresearch/dino#resnet-50-and-other-convnets-trainings
transform = DINOTransform(global_crop_scale=(0.14, 1), local_crop_scale=(0.05, 0.14))


class Scheduler:
    def __init__(self):
        self._step_count = 0

    def step(self) -> None:
        self._step_count += 1

    @property
    def current_step(self) -> int:
        return self._step_count

    def state_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_value(self, step: int) -> float:
        ...


class WarmupCosineScheduler(Scheduler):
    def __init__(
        self,
        start_value: float,
        end_value: float,
        max_steps: int,
        warmup_start_value: float = 0.0,
        warmup_end_value: float = 0.0,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.start_value = start_value
        self.end_value = end_value
        self.max_steps = max_steps
        self.warmup_start_value = warmup_start_value
        self.warmup_end_value = warmup_end_value
        self.warmup_steps = warmup_steps

    def get_value(self, step: int) -> float:
        if step < self.warmup_steps:
            return (
                self.warmup_start_value
                + (self.warmup_end_value - self.warmup_start_value)
                * step
                / self.warmup_steps
            )
        else:
            return cosine_schedule(
                step=step - self.warmup_steps,
                max_steps=self.max_steps - self.warmup_steps,
                start_value=self.start_value,
                end_value=self.end_value,
            )


momentum_scheduler = WarmupCosineScheduler(
    start_value=0.0,
    end_value=1.0,
    max_steps=1000,
)
momentum = momentum_scheduler.get_value(step=100)


def init_schedulers(optimizer):
    _validate_schedulers(optimizer)
    if _has_scheduler(optimizer):
        optimizer.register_optimizer_state_dict_post_hook(_state_dict_post_hook)
        optimizer.register_optimizer_load_state_dict_pre_hook(_load_state_dict_pre_hook)
        optimizer.register_optimizer_step_post_hook(_step_post_hook)


def _has_scheduler(optimizer):
    return any(
        isinstance(scheduler, Scheduler)
        for group in optimizer.param_groups
        for scheduler in group.values()
    )


def _validate_schedulers(optimizer):
    for group in optimizer.param_groups:
        for name, scheduler in group.items():
            if isinstance(scheduler, Scheduler):
                if not name.endswith("_scheduler"):
                    raise ValueError(f"Scheduler name must end with '_scheduler'.")


def _init_scheduler_values(optimizer):
    for group in optimizer.param_groups:
        for name in group.keys():
            scheduler = group[name]
            if isinstance(scheduler, Scheduler):
                value_name = name.rstrip("_scheduler")
                group.setdefault(f"initial_{value_name}", group[value_name])


def _state_dict_post_hook(optimizer, state_dict):
    for group in state_dict["param_groups"]:
        schedulers = {
            name: scheduler
            for name, scheduler in group.items()
            if isinstance(scheduler, Scheduler)
        }
        for name, scheduler in schedulers.items():
            group[name] = scheduler.state_dict()


def _load_state_dict_pre_hook(optimizer, state_dict):
    for group, group_state_dict in zip(
        optimizer.param_groups, state_dict["param_groups"]
    ):
        scheduler_state_dicts = {
            name: scheduler_state_dict
            for name, scheduler_state_dict in group_state_dict.items()
            if name.endswith("_scheduler")
        }
        for name in group.keys():
            if name in scheduler_state_dicts:
                group[name].load_state_dict(scheduler_state_dicts[name])
            elif name.endswith("_scheduler"):
                raise ValueError(f"Missing state_dict for scheduler '{name}'.")


def _step_post_hook(optimizer, *args, **kwargs) -> None:
    for group in optimizer.param_groups:
        for name, scheduler in group.items():
            if isinstance(scheduler, Scheduler):
                scheduler.step()
                value_name = name.rstrip("_scheduler")
                value = scheduler.get_value(step=scheduler.current_step)
                initial_value = group[f"initial_{value_name}"]
                group[value_name] = initial_value * value
