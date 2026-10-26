import math

import numpy as np
import pytest
import torch
from PIL import Image

from lightly.transforms import (
    BYOLTransform,
    DINOTransform,
    SimCLRTransform,
    SwaVTransform,
)
from lightly.utils import debug

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

BATCH_SIZE = 10
DIMENSION = 10


class TestDebug:
    def _generate_random_image(self, w: int, h: int, c: int):
        array = np.random.rand(h, w, c) * 255
        image = Image.fromarray(array.astype("uint8")).convert("RGB")
        return image

    def test_std_of_l2_normalized_collapsed(self):
        z = torch.ones(BATCH_SIZE, DIMENSION)  # collapsed output
        assert debug.std_of_l2_normalized(z) == 0.0

    def test_std_of_l2_normalized_uniform(self, eps: float = 1e-5):
        z = torch.eye(BATCH_SIZE)
        assert abs(debug.std_of_l2_normalized(z) - 1 / math.sqrt(z.shape[1])) <= eps

    def test_std_of_l2_normalized_raises(self):
        z = torch.zeros(BATCH_SIZE)
        with pytest.raises(ValueError):
            debug.std_of_l2_normalized(z)
        z = torch.zeros(BATCH_SIZE, BATCH_SIZE, DIMENSION)
        with pytest.raises(ValueError):
            debug.std_of_l2_normalized(z)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
    def test_plot_augmented_images(self):
        transform = SimCLRTransform(input_size=32)

        for n_images in range(2, 10):
            images = [self._generate_random_image(100, 100, 3) for _ in range(n_images)]
            fig = debug.plot_augmented_images(images, transform)
            assert fig is not None

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
    def test_plot_augmented_images_many_views(self):
        transform = DINOTransform(global_crop_size=32, local_crop_size=16)

        for n_images in range(1, 10):
            images = [self._generate_random_image(100, 100, 3) for _ in range(n_images)]
            fig = debug.plot_augmented_images(images, transform)
            assert fig is not None

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
    def test_plot_augmented_images_no_images(self):
        with pytest.raises(ValueError):
            debug.plot_augmented_images([], SimCLRTransform(input_size=32))

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
    def test_plot_augmented_images_invalid_transform(self):
        images = [self._generate_random_image(100, 100, 3)]
        with pytest.raises(ValueError):
            debug.plot_augmented_images(images, None)

    @pytest.mark.parametrize(
        "transform",
        [
            SimCLRTransform(input_size=32),
            DINOTransform(global_crop_size=32, local_crop_size=16),
            SwaVTransform(crop_sizes=(32, 16)),
            BYOLTransform(),
        ],
    )
    def test_generate_grid_of_augmented_images__returns_pil_images(self, transform):
        # ToTensor and Normalize must be skipped, otherwise the grid holds tensors
        # and plotting fails downstream.
        images = [self._generate_random_image(100, 100, 3) for _ in range(2)]

        grid = debug.generate_grid_of_augmented_images(images, transform)

        assert len(grid) == len(transform.transforms)
        for row in grid:
            assert len(row) == len(images)
            for image in row:
                assert isinstance(image, Image.Image)
