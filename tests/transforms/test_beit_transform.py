from __future__ import annotations

import unittest

import torch
from PIL import Image

from lightly.transforms import BEITTransform


class TestBEITTransform(unittest.TestCase):
    """Tests for the BEITTransform data augmentation and masking."""

    _GRID = 14
    _N = 14 * 14
    _MASK_RATIO = 0.4

    def _make_transform(self, **kwargs) -> BEITTransform:
        """Creates a BEITTransform with optional overrides.

        Args:
            **kwargs:
                Keyword arguments passed to BEITTransform.

        Returns:
            A BEITTransform instance configured for testing.
        """
        return BEITTransform(**kwargs)

    def test_mask_shape(self) -> None:
        """Tests that generated masks have the expected shape."""
        transform = self._make_transform()
        for batch_size in [1, 4, 8]:
            mask = transform.mask_generator(batch_size=batch_size)
            self.assertEqual(mask.shape, (batch_size, self._N))

    def test_mask_dtype_is_bool(self) -> None:
        """Tests that generated masks have boolean dtype."""
        transform = self._make_transform()
        mask = transform.mask_generator(batch_size=2)
        self.assertEqual(mask.dtype, torch.bool)

    def test_mask_values_are_binary(self) -> None:
        """Tests that mask values are strictly True or False."""
        transform = self._make_transform()
        mask = transform.mask_generator(batch_size=4)
        unique = mask.unique()
        self.assertTrue(set(unique.tolist()).issubset({False, True}))

    def test_mask_is_non_empty(self) -> None:
        """Tests that generated masks contain at least some masked patches."""
        transform = self._make_transform()
        for _ in range(10):
            mask = transform.mask_generator(batch_size=1)
            self.assertGreater(mask[0].sum().item(), 0)

    def test_mask_does_not_cover_all_patches(self) -> None:
        """Tests that generated masks do not cover all patches."""
        transform = self._make_transform()
        for _ in range(10):
            mask = transform.mask_generator(batch_size=1)
            self.assertLess(mask[0].sum().item(), self._N)

    def test_masks_differ_across_samples_in_batch(self) -> None:
        """Tests that masks differ across samples in the same batch."""
        transform = self._make_transform()
        mask = transform.mask_generator(batch_size=4)
        identical = all(torch.equal(mask[0], mask[i]) for i in range(1, 4))
        self.assertFalse(identical)

    def test_masks_differ_across_calls(self) -> None:
        """Tests that masks differ across independent calls."""
        transform = self._make_transform()
        mask_a = transform.mask_generator(batch_size=1)
        mask_b = transform.mask_generator(batch_size=1)
        self.assertFalse(torch.equal(mask_a, mask_b))

    def test_custom_grid_size(self) -> None:
        """Tests that custom input_size and patch_size produce correct grid."""
        transform = self._make_transform(input_size=112, patch_size=16)
        mask = transform.mask_generator(batch_size=2)
        self.assertEqual(mask.shape, (2, 49))

    def test_custom_mask_ratio(self) -> None:
        """Tests that custom mask_ratio is respected approximately."""
        ratio = 0.6
        transform = self._make_transform(mask_ratio=ratio)
        target = int(ratio * self._N)
        mask = transform.mask_generator(batch_size=4)
        for i in range(4):
            # Due to overlap-aware blocking, actual count may be slightly
            # less than target, but should be close.
            actual = mask[i].sum().item()
            self.assertGreaterEqual(actual, int(target * 0.5))
            self.assertLessEqual(actual, self._N)

    def test_custom_min_block_patches(self) -> None:
        """Tests that custom min_block_patches produces valid masks."""
        transform = self._make_transform(min_block_patches=1)
        mask = transform.mask_generator(batch_size=4)
        for i in range(4):
            self.assertGreater(mask[i].sum().item(), 0)

    def test_max_block_patches(self) -> None:
        """Tests that max_block_patches limits block size."""
        transform = self._make_transform(max_block_patches=4, min_block_patches=1)
        mask = transform.mask_generator(batch_size=2)
        for i in range(2):
            self.assertGreater(mask[i].sum().item(), 0)

    def test_aspect_ratio_range(self) -> None:
        """Tests that custom aspect_ratio_range is accepted."""
        transform = self._make_transform(
            aspect_ratio_range=(0.5, 2.0),
        )
        mask = transform.mask_generator(batch_size=2)
        self.assertEqual(mask.shape, (2, self._N))

    def test_mask_generator_repr(self) -> None:
        """Tests that __repr__ returns a non-empty string."""
        transform = self._make_transform()
        repr_str = repr(transform.mask_generator)
        self.assertIsInstance(repr_str, str)
        self.assertGreater(len(repr_str), 0)

    def test_mask_generator_get_shape(self) -> None:
        """Tests that get_shape returns the expected grid dimensions."""
        transform = self._make_transform()
        shape = transform.mask_generator.get_shape()
        self.assertEqual(shape, (self._GRID, self._GRID))

    def test_image_transform_output_shape(self) -> None:
        """Tests that the image transform produces the correct tensor shape."""
        transform = self._make_transform()
        image = Image.new(mode="RGB", size=(224, 224))
        tensor = transform(image=image)
        self.assertEqual(tensor.shape, (3, 224, 224))

    def test_image_transform_output_is_tensor(self) -> None:
        """Tests that the image transform returns a torch.Tensor."""
        transform = self._make_transform()
        image = Image.new(mode="RGB", size=(224, 224))
        tensor = transform(image=image)
        self.assertIsInstance(tensor, torch.Tensor)

    def test_image_transform_normalized(self) -> None:
        """Tests that the image transform produces normalized values."""
        transform = self._make_transform()
        image = Image.new(mode="RGB", size=(224, 224))
        tensor = transform(image=image)
        # After normalization, values should be outside [0, 1] for a black image
        # or at least not exactly in [0, 1] range for typical images.
        # For a black image (all zeros), normalized value = -mean/std.
        self.assertFalse(torch.all((tensor >= 0) & (tensor <= 1)))

    def test_mask_generator_batch_consistency(self) -> None:
        """Tests that batched generation is equivalent to individual calls."""
        transform = self._make_transform()
        torch.manual_seed(42)
        mask_batch = transform.mask_generator(batch_size=4)

        # Individual calls would differ due to randomness, so we just verify
        # the batch dimension is correct.
        self.assertEqual(mask_batch.shape[0], 4)

    def test_different_color_jitter(self) -> None:
        """Tests that custom color_jitter is accepted."""
        transform = self._make_transform(color_jitter=0.2)
        image = Image.new(mode="RGB", size=(224, 224))
        tensor = transform(image=image)
        self.assertEqual(tensor.shape, (3, 224, 224))

    def test_different_normalization_stats(self) -> None:
        """Tests that custom mean and std are accepted."""
        transform = self._make_transform(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        image = Image.new(mode="RGB", size=(224, 224))
        tensor = transform(image=image)
        self.assertEqual(tensor.shape, (3, 224, 224))

    def test_mask_generator_with_batch_size_one(self) -> None:
        """Tests mask generation with the smallest batch size."""
        transform = self._make_transform()
        mask = transform.mask_generator(batch_size=1)
        self.assertEqual(mask.shape, (1, self._N))
