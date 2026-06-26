import unittest

import torch

from lightly.models.modules import BEITEncoder
from lightly.models.modules.heads import MIMHead


class TestBEITEncoder(unittest.TestCase):
    _EMBED_DIM = 64
    _DEPTH = 2
    _NUM_HEADS = 4
    _IMG_SIZE = 32
    _PATCH_SIZE = 8

    def _make_encoder(self, **kwargs) -> BEITEncoder:
        return BEITEncoder(
            img_size=self._IMG_SIZE,
            patch_size=self._PATCH_SIZE,
            embed_dim=self._EMBED_DIM,
            depth=self._DEPTH,
            num_heads=self._NUM_HEADS,
            **kwargs,
        )

    def _make_mask(self, batch_size: int, mask_ratio: float = 0.4) -> torch.BoolTensor:
        n_patches = (self._IMG_SIZE // self._PATCH_SIZE) ** 2
        n_masked = int(n_patches * mask_ratio)
        mask = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        mask[:, :n_masked] = True
        return mask

    def test_output_shapes_no_mask(self) -> None:
        encoder = self._make_encoder()
        x = torch.randn(2, 3, self._IMG_SIZE, self._IMG_SIZE)
        out = encoder(x)
        n_patches = (self._IMG_SIZE // self._PATCH_SIZE) ** 2
        self.assertEqual(
            out["last_hidden_state"].shape, (2, n_patches + 1, self._EMBED_DIM)
        )
        self.assertEqual(out["patch_features"].shape, (2, n_patches, self._EMBED_DIM))
        self.assertEqual(out["cls_feature"].shape, (2, self._EMBED_DIM))

    def test_output_shapes_with_mask(self) -> None:
        encoder = self._make_encoder()
        x = torch.randn(2, 3, self._IMG_SIZE, self._IMG_SIZE)
        mask = self._make_mask(batch_size=2)
        out = encoder(x, bool_masked_pos=mask)
        n_patches = (self._IMG_SIZE // self._PATCH_SIZE) ** 2
        self.assertEqual(out["patch_features"].shape, (2, n_patches, self._EMBED_DIM))

    def test_all_attentions_returned(self) -> None:
        encoder = self._make_encoder()
        x = torch.randn(1, 3, self._IMG_SIZE, self._IMG_SIZE)
        out = encoder(x, return_all_attn=True)
        self.assertEqual(len(out["all_attentions"]), self._DEPTH)
        n_patches = (self._IMG_SIZE // self._PATCH_SIZE) ** 2
        for attn in out["all_attentions"]:
            self.assertEqual(
                attn.shape,
                (1, self._NUM_HEADS, n_patches + 1, n_patches + 1),
            )

    def test_no_attentions_by_default(self) -> None:
        encoder = self._make_encoder()
        x = torch.randn(1, 3, self._IMG_SIZE, self._IMG_SIZE)
        out = encoder(x, return_all_attn=False)
        self.assertEqual(out["all_attentions"], [])

    def test_masked_vs_unmasked_outputs_differ(self) -> None:
        torch.manual_seed(0)
        encoder = self._make_encoder()
        encoder.eval()
        x = torch.randn(1, 3, self._IMG_SIZE, self._IMG_SIZE)
        mask = self._make_mask(batch_size=1)
        with torch.no_grad():
            out_clean = encoder(x)["patch_features"]
            out_masked = encoder(x, bool_masked_pos=mask)["patch_features"]
        self.assertFalse(torch.allclose(out_clean, out_masked))

    def test_unmasked_positions_change_with_context(self) -> None:
        torch.manual_seed(0)
        encoder = self._make_encoder()
        encoder.eval()
        x = torch.randn(1, 3, self._IMG_SIZE, self._IMG_SIZE)
        mask = self._make_mask(batch_size=1, mask_ratio=0.5)
        with torch.no_grad():
            feats_clean = encoder(x)["patch_features"]
            feats_masked = encoder(x, bool_masked_pos=mask)["patch_features"]
        unmasked = ~mask[0]
        self.assertFalse(
            torch.allclose(feats_clean[:, unmasked], feats_masked[:, unmasked])
        )

    def test_gradient_flows_through_encoder(self) -> None:
        encoder = self._make_encoder()
        x = torch.randn(1, 3, self._IMG_SIZE, self._IMG_SIZE, requires_grad=True)
        mask = self._make_mask(batch_size=1)
        out = encoder(x, bool_masked_pos=mask)
        out["patch_features"].sum().backward()
        self.assertIsNotNone(x.grad)

    def test_mask_token_is_parameter(self) -> None:
        encoder = self._make_encoder()
        self.assertIsInstance(encoder.mask_token, torch.nn.Parameter)
        self.assertEqual(encoder.mask_token.shape, (1, 1, self._EMBED_DIM))

    def test_cls_token_is_parameter(self) -> None:
        encoder = self._make_encoder()
        self.assertIsInstance(encoder.cls_token, torch.nn.Parameter)
        self.assertEqual(encoder.cls_token.shape, (1, 1, self._EMBED_DIM))

    def test_consistent_across_batch_sizes(self) -> None:
        encoder = self._make_encoder()
        encoder.eval()
        x = torch.randn(1, 3, self._IMG_SIZE, self._IMG_SIZE)
        mask_1 = self._make_mask(batch_size=1)
        mask_2 = mask_1.expand(2, -1)
        with torch.no_grad():
            out_1 = encoder(x, bool_masked_pos=mask_1)["patch_features"]
            out_2 = encoder(x.expand(2, -1, -1, -1), bool_masked_pos=mask_2)[
                "patch_features"
            ]
        self.assertTrue(torch.allclose(out_1, out_2[:1], atol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available")
    def test_forward_pass_cuda(self) -> None:
        encoder = self._make_encoder().cuda()
        x = torch.randn(2, 3, self._IMG_SIZE, self._IMG_SIZE).cuda()
        mask = self._make_mask(batch_size=2).cuda()
        out = encoder(x, bool_masked_pos=mask)
        self.assertTrue(out["patch_features"].is_cuda)


class TestMIMHead(unittest.TestCase):
    _EMBED_DIM = 64
    _VOCAB_SIZE = 512

    def _make_head(self) -> MIMHead:
        return MIMHead(embed_dim=self._EMBED_DIM, vocab_size=self._VOCAB_SIZE)

    def test_output_shape(self) -> None:
        head = self._make_head()
        features = torch.randn(2, 16, self._EMBED_DIM)
        logits = head(features)
        self.assertEqual(logits.shape, (2, 16, self._VOCAB_SIZE))

    def test_gradient_flows(self) -> None:
        head = self._make_head()
        features = torch.randn(2, 16, self._EMBED_DIM, requires_grad=True)
        logits = head(features)
        logits.sum().backward()
        self.assertIsNotNone(features.grad)

    def test_single_token(self) -> None:
        head = self._make_head()
        features = torch.randn(1, 1, self._EMBED_DIM)
        logits = head(features)
        self.assertEqual(logits.shape, (1, 1, self._VOCAB_SIZE))

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available")
    def test_forward_pass_cuda(self) -> None:
        head = self._make_head().cuda()
        features = torch.randn(2, 16, self._EMBED_DIM).cuda()
        logits = head(features)
        self.assertTrue(logits.is_cuda)
        self.assertEqual(logits.shape, (2, 16, self._VOCAB_SIZE))
