import unittest
import torch
import math
import numpy as np
from PIL import Image

from lightly.utils import debug
from lightly.data import collate

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

BATCH_SIZE = 10
DIMENSION = 10


class TestDebug(unittest.TestCase):

    def _generate_random_image(self, w: int, h: int, c: int):
        array = np.random.rand(h, w, c) * 255
        image = Image.fromarray(array.astype('uint8')).convert('RGB')
        return image

    def test_std_of_l2_normalized_collapsed(self):
        z = torch.ones(BATCH_SIZE, DIMENSION) # collapsed output
        self.assertEqual(debug.std_of_l2_normalized(z), 0.0)

    def test_std_of_l2_normalized_uniform(self, eps: float = 1e-5):
        z = torch.eye(BATCH_SIZE)
        self.assertLessEqual(
            abs(debug.std_of_l2_normalized(z) - 1 / math.sqrt(z.shape[1])),
            eps,
        )

    def test_std_of_l2_normalized_raises(self):
        z = torch.zeros(BATCH_SIZE)
        with self.assertRaises(ValueError):
            debug.std_of_l2_normalized(z)
        z = torch.zeros(BATCH_SIZE, BATCH_SIZE, DIMENSION)
        with self.assertRaises(ValueError):
            debug.std_of_l2_normalized(z)

    @unittest.skipUnless(MATPLOTLIB_AVAILABLE, "Matplotlib not installed")
    def test_plot_augmented_images_image_collate_function(self):

        # simclr collate function is a subclass of the image collate function
        collate_function = collate.SimCLRCollateFunction()

        for n_images in range(2, 10):
            with self.subTest():
                images = [
                    self._generate_random_image(100, 100, 3)
                    for _ in range(n_images)
                ]
                fig = debug.plot_augmented_images(images, collate_function)
                self.assertIsNotNone(fig)

    @unittest.skipUnless(MATPLOTLIB_AVAILABLE, "Matplotlib not installed")
    def test_plot_augmented_images_multi_view_collate_function(self):

        # dion collate function is a subclass of the multi view collate function
        collate_function = collate.DINOCollateFunction()

        for n_images in range(1, 10):
            with self.subTest():
                images = [
                    self._generate_random_image(100, 100, 3)
                    for _ in range(n_images)
                ]
                fig = debug.plot_augmented_images(images, collate_function)
                self.assertIsNotNone(fig)

    @unittest.skipUnless(MATPLOTLIB_AVAILABLE, "Matplotlib not installed")
    def test_plot_augmented_images_no_images(self):
        collate_function = collate.SimCLRCollateFunction()
        with self.assertRaises(ValueError):
            debug.plot_augmented_images([], collate_function)

    @unittest.skipUnless(MATPLOTLIB_AVAILABLE, "Matplotlib not installed")
    def test_plot_augmented_images_invalid_collate_function(self):
        images = [self._generate_random_image(100, 100, 3)]
        with self.assertRaises(ValueError):
            debug.plot_augmented_images(images, None)


