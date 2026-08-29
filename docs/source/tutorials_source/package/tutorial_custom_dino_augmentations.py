""".. _lightly-custom-dino-augmentations-tutorial-9:

Tutorial 9: Customize DINO Views with AlbumentationsX
=====================================================

DINO learns by comparing several independently augmented views of the same image.
Lightly's :class:`~lightly.transforms.dino_transform.DINOTransform` provides the
standard policy, but `Lightly issue #1814
<https://github.com/lightly-ai/lightly/issues/1814>`_ shows that changing an
unexposed parameter such as the aspect-ratio range of ``RandomResizedCrop`` otherwise
requires copying most of that transform.

This tutorial keeps Lightly's current multi-view and training contracts while moving
each image policy into AlbumentationsX. You will learn how to:

- preserve DINO's two global-view roles and repeated local-view role;
- change crop ratio, scale, or photometric operations in one policy;
- pass the policies to Lightly's
  :class:`~lightly.transforms.multi_view_transform.MultiViewTransform`;
- serialize the policies with the experiment configuration.

Prerequisites
-------------

This optional integration requires Python 3.10 or newer. Install the PyTorch build
for your CPU, CUDA, or MPS environment, then install Lightly and AlbumentationsX:

.. code-block:: console

    python -m pip install lightly
    python -m pip install "albumentationsx[headless]>=2.4.3"

AlbumentationsX is distributed under the AGPL-3.0-only license and remains separate
from Lightly's dependencies. It imports as ``albumentations``. The ``headless`` extra
uses OpenCV without desktop GUI components; use the ``gui`` extra if your application
needs them.
"""

# %%
# Imports
# -------
from __future__ import annotations

import json

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from lightly.transforms.multi_view_transform import MultiViewTransform

# %%
# Define one policy per DINO view role
# ------------------------------------
#
# DINO does not apply one sampled crop to every view. Each view is an independent
# realization of a role-specific policy:
#
# - ``global_view_0`` always applies Gaussian blur;
# - ``global_view_1`` applies blur with probability 0.1 and solarization with
#   probability 0.2; and
# - every local view uses the same small-crop policy, with blur probability 0.5.
#
# The two global policies share crop size and scale, but remain separate because their
# photometric probabilities differ. ``MultiViewTransform`` calls every list entry once.
# Repeating one local policy six times therefore samples six independent local views.

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_dino_view(
    *,
    size: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    blur_probability: float,
    solarization_probability: float,
) -> A.Compose:
    """Builds one DINO view policy with explicit crop and photometric settings."""
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(size, size),
                scale=scale,
                ratio=ratio,
                interpolation=cv2.INTER_CUBIC,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness_range=(0.6, 1.4),
                contrast_range=(0.6, 1.4),
                saturation_range=(0.8, 1.2),
                hue_range=(-0.1, 0.1),
                p=0.8,
            ),
            A.ToGray(num_output_channels=3, p=0.2),
            A.GaussianBlur(
                sigma_range=(0.1, 2.0),
                blur_range=(0, 0),
                p=blur_probability,
            ),
            A.Solarize(
                threshold_range=(0.5, 0.5),
                p=solarization_probability,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            A.ToTensorV2(),
        ],
    )


# ``ratio`` is explicit, so changing the experiment no longer requires copying
# Lightly's complete ``DINOTransform``. The values below deliberately widen the
# default 3:4-to-4:3 range to demonstrate the customization point.
global_view_0 = make_dino_view(
    size=224,
    scale=(0.4, 1.0),
    ratio=(0.7, 1.4),
    blur_probability=1.0,
    solarization_probability=0.0,
)
global_view_1 = make_dino_view(
    size=224,
    scale=(0.4, 1.0),
    ratio=(0.7, 1.4),
    blur_probability=0.1,
    solarization_probability=0.2,
)
local_view = make_dino_view(
    size=96,
    scale=(0.05, 0.4),
    ratio=(0.7, 1.4),
    blur_probability=0.5,
    solarization_probability=0.0,
)

# %%
# Adapt PIL input to the Lightly transform contract
# -------------------------------------------------
#
# ``LightlyDataset`` normally loads an RGB PIL image. AlbumentationsX accepts an
# HWC NumPy array and returns a dictionary, so this adapter exposes the callable
# interface expected by ``MultiViewTransform``. ``A.ToTensorV2`` makes each output a
# CHW PyTorch tensor ready for the DINO model.


class AlbumentationsView:
    """Adapts an AlbumentationsX image pipeline to Lightly's view callable."""

    def __init__(self, pipeline: A.Compose) -> None:
        self.pipeline = pipeline

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image_array = np.asarray(image.convert("RGB"))
        transformed = self.pipeline(image=image_array)
        return transformed["image"]


n_local_views = 6
local_transform = AlbumentationsView(local_view)
transform = MultiViewTransform(
    transforms=[
        AlbumentationsView(global_view_0),
        AlbumentationsView(global_view_1),
        *[local_transform] * n_local_views,
    ]
)

# %%
# Inspect the multi-view output
# -----------------------------
#
# The synthetic image below makes spatially different crops easy to recognize and
# keeps this tutorial runnable without downloading a dataset. The transform returns
# two 224x224 global views followed by six 96x96 local views.

height, width = 320, 480
row, column = np.mgrid[:height, :width]
image_array = np.stack(
    [
        255 * column / (width - 1),
        255 * row / (height - 1),
        np.where((row // 40 + column // 40) % 2 == 0, 45, 210),
    ],
    axis=-1,
).astype(np.uint8)
image_array[50:145, 60:205] = (235, 65, 65)
image_array[180:295, 285:440] = (45, 215, 95)
image = Image.fromarray(image_array)

views = transform(image)
assert len(views) == 2 + n_local_views
assert [tuple(view.shape) for view in views[:2]] == [(3, 224, 224)] * 2
assert [tuple(view.shape) for view in views[2:]] == [(3, 96, 96)] * n_local_views


def image_for_plot(view: torch.Tensor) -> np.ndarray:
    """Reverses ImageNet normalization and converts a view from CHW to HWC."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=view.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=view.dtype).view(3, 1, 1)
    image_tensor = (view * std + mean).clamp(0, 1)
    return image_tensor.permute(1, 2, 0).numpy()


figure, axes = plt.subplots(3, 3, figsize=(9, 9))
flat_axes = axes.ravel()
flat_axes[0].imshow(image)
flat_axes[0].set_title("Input")
for index, view in enumerate(views):
    flat_axes[index + 1].imshow(image_for_plot(view))
    role = f"Global {index}" if index < 2 else f"Local {index - 2}"
    flat_axes[index + 1].set_title(role)
for axis in flat_axes:
    axis.set_axis_off()
figure.tight_layout()

# %%
# Use the normal Lightly training path
# ------------------------------------
#
# ``transform`` has the same current input/output boundary as ``DINOTransform``:
# one PIL image in, then two global tensors followed by the local tensors. Attach it
# to a dataset exactly as you would attach Lightly's built-in transform:
#
# .. code-block:: python
#
#    from lightly.data import LightlyDataset
#
#    dataset = LightlyDataset(input_dir="path/to/images", transform=transform)
#    dataloader = torch.utils.data.DataLoader(
#        dataset,
#        batch_size=64,
#        shuffle=True,
#        drop_last=True,
#        num_workers=8,
#    )
#
# In the DINO training loop, the teacher receives ``views[:2]`` and the student
# receives every view. No model or loss changes are required.

# %%
# Serialize the three policies
# ----------------------------
#
# Serialization stores the configured transform graphs, including the explicit crop
# ratios and role-specific probabilities. It does not store a particular random
# realization.

view_pipelines = {
    "global_view_0": global_view_0,
    "global_view_1": global_view_1,
    "local_view": local_view,
}
serialized_policies = {
    name: A.to_dict(pipeline) for name, pipeline in view_pipelines.items()
}
serialized_json = json.dumps(serialized_policies, indent=2)
restored_global_view_1 = A.from_dict(json.loads(serialized_json)["global_view_1"])
assert A.to_dict(restored_global_view_1) == serialized_policies["global_view_1"]
print(f"Serialized policies: {', '.join(serialized_policies)}")

# Persist ``serialized_json`` with the rest of your experiment configuration. Restore
# any policy with ``A.from_dict`` before constructing ``AlbumentationsView``.

# %%
# Keep Lightly's training workflow
# --------------------------------
#
# Lightly still owns the multi-view ordering, dataset boundary, DINO model, loss, and
# training loop. AlbumentationsX owns only the three image policies. This separation is
# useful when an experiment needs a crop ratio or operation that ``DINOTransform`` does
# not expose. The policies above preserve DINO's view sizes, scale ranges, and
# role-specific photometric probabilities while intentionally changing crop ratio.
# OpenCV and torchvision can still produce different pixels for nominally equivalent
# operations. Treat a policy change as an experiment and evaluate the learned
# representation on the intended downstream task.
