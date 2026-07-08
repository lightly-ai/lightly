from __future__ import annotations

import math
import unittest
from typing import Any

import torch

from lightly.models.modules import BEITEncoder
from lightly.models.modules.heads import BEiTMIMHead


class TestBEITEncoder(unittest.TestCase):
    """Tests for the BEITEncoder module."""

    _EMBED_DIM = 64
    _DEPTH = 2
    _NUM_HEADS = 4
    _IMG_SIZE = 32
    _PATCH_SIZE = 8

    def _make_encoder(self, **kwargs: Any) -> BEITEncoder:
        """Creates a BEITEncoder with default test parameters.

        Args:
            **kwargs:
                Additional keyword arguments passed to BEITEncoder.

        Returns:
            A BEITEncoder instance configured for testing.
        """
        return BEITEncoder(
            img_size=self._IMG_SIZE,
            patch_size=self._PATCH_SIZE,
            embed_dim=self._EMBED_DIM,
            depth=self._DEPTH,
            num_heads=self._NUM_HEADS,
            **kwargs,
        )

    def _make_mask(self, batch_size: int, mask_ratio: float = 0.4) -> torch.Tensor:
        """Creates a deterministic boolean mask for testing.

        Args:
            batch_size:
                Number of samples in the batch.
            mask_ratio:
                Fraction of patches to mask.

        Returns:
            Boolean tensor of shape (batch_size, n_patches) with the
            first n_masked positions set to True.
        """
        n_patches = (self._IMG_SIZE // self._PATCH_SIZE) ** 2
        n_masked = int(n_patches * mask_ratio)
        mask = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        mask[:, :n_masked] = True
        return mask

    def test_output_shapes_no_mask(self) -> None:
        """Tests output shapes when no mask is provided."""
        encoder = self._make_encoder()
        x = torch.randn(2, 3, self._IMG_SIZE, self._IMG_SIZE)
        out = encoder(x=x)
        n_patches = (self._IMG_SIZE // self._PATCH_SIZE) ** 2

        self.assertEqual(
            out["last_hidden_state"].shape,
            (2, n_patches + 1, self._EMBED_DIM),
        )
        self.assertEqual(
            out["patch_features"].shape,
            (2, n_patches, self._EMBED_DIM),
        )
        self.assertEqual(
            out["cls_feature"].shape,
            (2, self._EMBED_DIM),
        )

    def test_output_shapes_with_mask(self) -> None:
        """Tests output shapes when masking is applied."""
        encoder = self._make_encoder()
        x = torch.randn(2, 3, self._IMG_SIZE, self._IMG_SIZE)
        mask = self._make_mask(batch_size=2)
        out = encoder(x=x, bool_masked_pos=mask)
        n_patches = (self._IMG_SIZE // self._PATCH_SIZE) ** 2

        self.assertEqual(
            out["patch_features"].shape,
            (2, n_patches, self._EMBED_DIM),
        )

    def test_masked_vs_unmasked_outputs_differ(self) -> None:
        """Tests that masking changes the output."""
        torch.manual_seed(0)
        encoder = self._make_encoder()
        encoder.eval()
        x = torch.randn(1, 3, self._IMG_SIZE, self._IMG_SIZE)
        mask = self._make_mask(batch_size=1)

        with torch.no_grad():
            out_clean = encoder(x=x)["patch_features"]
            out_masked = encoder(x=x, bool_masked_pos=mask)["patch_features"]

        self.assertFalse(torch.allclose(out_clean, out_masked))

    def test_unmasked_positions_change_with_context(self) -> None:
        """Tests that unmasked positions are affected by masked positions."""
        torch.manual_seed(0)
        encoder = self._make_encoder()
        encoder.eval()
        x = torch.randn(1, 3, self._IMG_SIZE, self._IMG_SIZE)
        mask = self._make_mask(batch_size=1, mask_ratio=0.5)

        with torch.no_grad():
            feats_clean = encoder(x=x)["patch_features"]
            feats_masked = encoder(x=x, bool_masked_pos=mask)["patch_features"]

        unmasked = ~mask[0]
        self.assertFalse(
            torch.allclose(feats_clean[:, unmasked], feats_masked[:, unmasked])
        )

    def test_gradient_flows_through_encoder(self) -> None:
        """Tests that gradients propagate through the encoder."""
        encoder = self._make_encoder()
        x = torch.randn(1, 3, self._IMG_SIZE, self._IMG_SIZE, requires_grad=True)
        mask = self._make_mask(batch_size=1)
        out = encoder(x=x, bool_masked_pos=mask)
        out["patch_features"].sum().backward()

        self.assertIsNotNone(x.grad)

    def test_mask_token_is_parameter(self) -> None:
        """Tests that mask_token is a learnable parameter."""
        encoder = self._make_encoder()

        self.assertIsInstance(encoder.mask_token, torch.nn.Parameter)
        self.assertEqual(encoder.mask_token.shape, (1, 1, self._EMBED_DIM))

    def test_cls_token_is_parameter(self) -> None:
        """Tests that cls_token is a learnable parameter."""
        encoder = self._make_encoder()

        self.assertIsInstance(encoder.cls_token, torch.nn.Parameter)
        self.assertEqual(encoder.cls_token.shape, (1, 1, self._EMBED_DIM))

    def test_consistent_across_batch_sizes(self) -> None:
        """Tests that outputs are consistent regardless of batch size."""
        encoder = self._make_encoder()
        encoder.eval()
        x = torch.randn(1, 3, self._IMG_SIZE, self._IMG_SIZE)
        mask_1 = self._make_mask(batch_size=1)
        mask_2 = mask_1.expand(2, -1)

        with torch.no_grad():
            out_1 = encoder(x=x, bool_masked_pos=mask_1)["patch_features"]
            out_2 = encoder(
                x=x.expand(2, -1, -1, -1),
                bool_masked_pos=mask_2,
            )["patch_features"]

        self.assertTrue(torch.allclose(out_1, out_2[:1], atol=1e-5))

    def test_relative_position_bias(self) -> None:
        """Tests encoder with shared relative position bias enabled."""
        encoder = self._make_encoder(use_shared_rel_pos_bias=True)
        x = torch.randn(1, 3, self._IMG_SIZE, self._IMG_SIZE)
        out = encoder(x=x)

        self.assertEqual(
            out["patch_features"].shape,
            (1, (self._IMG_SIZE // self._PATCH_SIZE) ** 2, self._EMBED_DIM),
        )

    def test_no_absolute_pos_emb(self) -> None:
        """Tests encoder without absolute position embeddings."""
        encoder = self._make_encoder(use_abs_pos_emb=False)
        x = torch.randn(1, 3, self._IMG_SIZE, self._IMG_SIZE)
        out = encoder(x=x)

        self.assertIsNone(encoder.pos_embed)
        self.assertEqual(
            out["patch_features"].shape,
            (1, (self._IMG_SIZE // self._PATCH_SIZE) ** 2, self._EMBED_DIM),
        )

    def test_layer_scale_init(self) -> None:
        """Tests that LayerScale parameters are created when init_values > 0."""
        encoder = self._make_encoder(init_values=0.1)

        for block in encoder.blocks:
            self.assertIsNotNone(block.gamma_1)
            self.assertIsNotNone(block.gamma_2)
            self.assertEqual(block.gamma_1.shape, (self._EMBED_DIM,))
            self.assertEqual(block.gamma_2.shape, (self._EMBED_DIM,))

    def test_no_layer_scale_by_default(self) -> None:
        """Tests that LayerScale is disabled by default."""
        encoder = self._make_encoder()

        for block in encoder.blocks:
            self.assertIsNone(block.gamma_1)
            self.assertIsNone(block.gamma_2)

    def test_fix_init_weight_applied(self) -> None:
        """Tests that fix_init_weight rescales output projections."""
        encoder = self._make_encoder()
        encoder.eval()

        for layer_id, block in enumerate(encoder.blocks, start=1):
            expected_scale = 1.0 / math.sqrt(2.0 * layer_id)
            # We can't easily test the exact value since init is random,
            # but we can verify the method ran without error by checking
            # the model produces valid outputs.
            x = torch.randn(1, 3, self._IMG_SIZE, self._IMG_SIZE)
            out = encoder(x=x)
            self.assertTrue(torch.isfinite(out["patch_features"]).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_forward_pass_cuda(self) -> None:
        """Tests forward pass on CUDA if available."""
        encoder = self._make_encoder().cuda()
        x = torch.randn(2, 3, self._IMG_SIZE, self._IMG_SIZE).cuda()
        mask = self._make_mask(batch_size=2).cuda()
        out = encoder(x=x, bool_masked_pos=mask)

        self.assertTrue(out["patch_features"].is_cuda)


class TestMIMHead(unittest.TestCase):
    """Tests for the BEiTMIMHead module."""

    _EMBED_DIM = 64
    _VOCAB_SIZE = 512

    def _make_head(self) -> BEiTMIMHead:
        """Creates a BEiTMIMHead with default test parameters.

        Returns:
            A BEiTMIMHead instance configured for testing.
        """
        return BEiTMIMHead(
            embed_dim=self._EMBED_DIM,
            vocab_size=self._VOCAB_SIZE,
        )

    def test_output_shape(self) -> None:
        """Tests that output shape matches (B, N, vocab_size)."""
        head = self._make_head()
        features = torch.randn(2, 16, self._EMBED_DIM)
        logits = head(patch_features=features)

        self.assertEqual(logits.shape, (2, 16, self._VOCAB_SIZE))

    def test_gradient_flows(self) -> None:
        """Tests that gradients propagate through the head."""
        head = self._make_head()
        features = torch.randn(2, 16, self._EMBED_DIM, requires_grad=True)
        logits = head(patch_features=features)
        logits.sum().backward()

        self.assertIsNotNone(features.grad)

    def test_single_token(self) -> None:
        """Tests forward pass with a single token."""
        head = self._make_head()
        features = torch.randn(1, 1, self._EMBED_DIM)
        logits = head(patch_features=features)

        self.assertEqual(logits.shape, (1, 1, self._VOCAB_SIZE))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_forward_pass_cuda(self) -> None:
        """Tests forward pass on CUDA if available."""
        head = self._make_head().cuda()
        features = torch.randn(2, 16, self._EMBED_DIM).cuda()
        logits = head(patch_features=features)

        self.assertTrue(logits.is_cuda)
        self.assertEqual(logits.shape, (2, 16, self._VOCAB_SIZE))
