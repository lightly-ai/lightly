import unittest

import torch

from lightly.models import utils
from lightly.utils import dependency

if dependency.timm_vit_available():
    import timm

    from lightly.models.modules.masked_autoencoder_timm import MAEDecoder
    from lightly.models.modules.masked_vision_transformer_timm import (
        MaskedVisionTransformerTIMM,
    )

if dependency.torchvision_vit_available():
    import torchvision

    from lightly.models.modules import MAEBackbone, MAEDecoder, MAEEncoder


@unittest.skipUnless(dependency.timm_vit_available(), "TIMM is not available")
class TestMAEBackbone(unittest.TestCase):
    def _vit(self):
        return timm.models.vision_transformer.vit_base_patch32_224()

    def test_from_vit(self):
        MaskedVisionTransformerTIMM(vit=self._vit())

    def _test_forward(self, device, batch_size=8, seed=0):
        torch.manual_seed(seed)
        vit = self._vit()
        backbone = MaskedVisionTransformerTIMM(vit=vit).to(device)
        images = torch.rand(
            batch_size, 3, vit.patch_embed.img_size[0], vit.patch_embed.img_size[0]
        ).to(device)
        _idx_keep, _ = utils.random_token_mask(
            size=(batch_size, backbone.sequence_length),
            device=device,
        )
        for idx_keep in [None, _idx_keep]:
            with self.subTest(idx_keep=idx_keep):
                class_tokens = backbone(images=images, idx_keep=idx_keep)

                # output shape must be correct
                expected_shape = [batch_size, vit.embed_dim]
                self.assertListEqual(list(class_tokens.shape), expected_shape)

                # output must have reasonable numbers
                self.assertTrue(torch.all(torch.not_equal(class_tokens, torch.inf)))

    def test_forward(self):
        self._test_forward(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available.")
    def test_forward_cuda(self):
        self._test_forward(torch.device("cuda"))

    def test_images_to_tokens(self) -> None:
        torch.manual_seed(0)
        vit = self._vit()
        backbone = MaskedVisionTransformerTIMM(vit=vit)
        images = torch.rand(2, 3, 224, 224)
        assert torch.all(
            vit.patch_embed(images) == backbone.images_to_tokens(images=images)
        )


@unittest.skipUnless(dependency.timm_vit_available(), "TIMM is not available")
class TestMAEDecoder(unittest.TestCase):
    def test_init(self):
        MAEDecoder(
            num_patches=49,
            patch_size=32,
            embed_dim=128,
            decoder_embed_dim=256,
            decoder_depth=2,
            decoder_num_heads=4,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

    def _test_forward(self, device, batch_size=8, seed=0):
        torch.manual_seed(seed)
        seq_length = 50
        embed_input_dim = 128
        patch_size = 32
        out_dim = 3 * patch_size**2
        decoder = MAEDecoder(
            num_patches=49,
            patch_size=32,
            embed_dim=embed_input_dim,
            decoder_embed_dim=256,
            decoder_depth=2,
            decoder_num_heads=4,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        ).to(device)
        tokens = torch.rand(batch_size, seq_length, embed_input_dim).to(device)
        predictions = decoder(tokens)

        # output shape must be correct
        expected_shape = [batch_size, seq_length, out_dim]
        self.assertListEqual(list(predictions.shape), expected_shape)

        # output must have reasonable numbers
        self.assertTrue(torch.all(torch.not_equal(predictions, torch.inf)))

    def test_forward(self):
        self._test_forward(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available.")
    def test_forward_cuda(self):
        self._test_forward(torch.device("cuda"))


@unittest.skipUnless(
    dependency.torchvision_vit_available(), "Torchvision ViT not available"
)
class TestMAEEncoder(unittest.TestCase):
    def _vit(self):
        return torchvision.models.vision_transformer.vit_b_32(progress=False)

    def test_from_vit(self):
        MAEEncoder.from_vit_encoder(self._vit().encoder)

    def _test_forward(self, device, batch_size=8, seed=0):
        torch.manual_seed(seed)
        vit = self._vit()
        encoder = MAEEncoder.from_vit_encoder(vit.encoder).to(device)
        tokens = torch.rand(batch_size, vit.seq_length, vit.hidden_dim).to(device)
        _idx_keep, _ = utils.random_token_mask(
            size=(batch_size, vit.seq_length),
            device=device,
        )
        for idx_keep in [None, _idx_keep]:
            with self.subTest(idx_keep=idx_keep):
                out = encoder(tokens, idx_keep)

                # output shape must be correct
                expected_shape = list(tokens.shape)
                if idx_keep is not None:
                    expected_shape[1] = idx_keep.shape[1]
                self.assertListEqual(list(out.shape), expected_shape)

                # output must have reasonable numbers
                self.assertTrue(torch.all(torch.not_equal(out, torch.inf)))

    def test_forward(self):
        self._test_forward(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available.")
    def test_forward_cuda(self):
        self._test_forward(torch.device("cuda"))


@unittest.skipUnless(
    dependency.torchvision_vit_available(), "Torchvision ViT not available"
)
class TestMAEBackbone(unittest.TestCase):
    def _vit(self):
        return torchvision.models.vision_transformer.vit_b_32(progress=False)

    def test_from_vit(self):
        MAEBackbone.from_vit(self._vit())

    def _test_forward(self, device, batch_size=8, seed=0):
        torch.manual_seed(seed)
        vit = self._vit()
        backbone = MAEBackbone.from_vit(vit).to(device)
        images = torch.rand(batch_size, 3, vit.image_size, vit.image_size).to(device)
        _idx_keep, _ = utils.random_token_mask(
            size=(batch_size, vit.seq_length),
            device=device,
        )
        for idx_keep in [None, _idx_keep]:
            with self.subTest(idx_keep=idx_keep):
                class_tokens = backbone(images, idx_keep)

                # output shape must be correct
                expected_shape = [batch_size, vit.hidden_dim]
                self.assertListEqual(list(class_tokens.shape), expected_shape)

                # output must have reasonable numbers
                self.assertTrue(torch.all(torch.not_equal(class_tokens, torch.inf)))

    def test_forward(self):
        self._test_forward(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available.")
    def test_forward_cuda(self):
        self._test_forward(torch.device("cuda"))


@unittest.skipUnless(
    dependency.torchvision_vit_available(), "Torchvision ViT not available"
)
class TestMAEDecoder(unittest.TestCase):
    def test_init(self):
        MAEDecoder(
            seq_length=50,
            num_layers=2,
            num_heads=4,
            embed_input_dim=128,
            hidden_dim=256,
            mlp_dim=256 * 4,
            out_dim=3 * 32**2,
        )

    def _test_forward(self, device, batch_size=8, seed=0):
        torch.manual_seed(seed)
        seq_length = 50
        embed_input_dim = 128
        out_dim = 3 * 32**2
        decoder = MAEDecoder(
            seq_length=seq_length,
            num_layers=2,
            num_heads=4,
            embed_input_dim=embed_input_dim,
            hidden_dim=256,
            mlp_dim=256 * 4,
            out_dim=out_dim,
        ).to(device)
        tokens = torch.rand(batch_size, seq_length, embed_input_dim).to(device)
        predictions = decoder(tokens)

        # output shape must be correct
        expected_shape = [batch_size, seq_length, out_dim]
        self.assertListEqual(list(predictions.shape), expected_shape)

        # output must have reasonable numbers
        self.assertTrue(torch.all(torch.not_equal(predictions, torch.inf)))

    def test_forward(self):
        self._test_forward(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available.")
    def test_forward_cuda(self):
        self._test_forward(torch.device("cuda"))
