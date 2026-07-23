from __future__ import annotations

import pytest
import torch
import torchvision

from lightly.models import utils
from lightly.utils import dependency

if dependency.torchvision_vit_available():
    from torchvision.models.vision_transformer import VisionTransformer

    from lightly.models.modules import MAEBackbone, MAEDecoder, MAEEncoder


@pytest.mark.skipif(
    not dependency.torchvision_vit_available(),
    reason="Torchvision ViT not available",
)
class TestMAEEncoder:
    def _vit(self) -> VisionTransformer:
        return torchvision.models.vision_transformer.vit_b_32(progress=False)

    def test_from_vit(self) -> None:
        MAEEncoder.from_vit_encoder(self._vit().encoder)

    def _test_forward(
        self,
        device: torch.device,
        use_mask: bool,
        batch_size: int = 8,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        vit = self._vit()
        encoder = MAEEncoder.from_vit_encoder(vit.encoder).to(device)
        tokens = torch.rand(batch_size, vit.seq_length, vit.hidden_dim).to(device)
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, vit.seq_length),
            device=device,
        )
        if not use_mask:
            idx_keep = None
        out = encoder(tokens, idx_keep)

        # output shape must be correct
        expected_shape = list(tokens.shape)
        if idx_keep is not None:
            expected_shape[1] = idx_keep.shape[1]
        assert list(out.shape) == expected_shape

        # output must have reasonable numbers
        assert torch.all(torch.not_equal(out, torch.inf))

    @pytest.mark.parametrize("use_mask", [False, True])
    def test_forward(self, use_mask: bool) -> None:
        self._test_forward(device=torch.device("cpu"), use_mask=use_mask)

    @pytest.mark.parametrize("use_mask", [False, True])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda not available.")
    def test_forward_cuda(self, use_mask: bool) -> None:
        self._test_forward(torch.device("cuda"), use_mask=use_mask)


@pytest.mark.skipif(
    not dependency.torchvision_vit_available(),
    reason="Torchvision ViT not available",
)
class TestMAEBackbone:
    def _vit(self) -> torchvision.models.vision_transformer.VisionTransformer:
        return torchvision.models.vision_transformer.vit_b_32(progress=False)

    def test_from_vit(self) -> None:
        MAEBackbone.from_vit(self._vit())

    def _test_forward(
        self,
        device: torch.device,
        use_mask: bool,
        batch_size: int = 8,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        vit = self._vit()
        backbone = MAEBackbone.from_vit(vit).to(device)
        images = torch.rand(batch_size, 3, vit.image_size, vit.image_size).to(device)
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, vit.seq_length),
            device=device,
        )
        if not use_mask:
            idx_keep = None
        class_tokens = backbone(images, idx_keep)

        # output shape must be correct
        expected_shape = [batch_size, vit.hidden_dim]
        assert list(class_tokens.shape) == expected_shape

        # output must have reasonable numbers
        assert torch.all(torch.not_equal(class_tokens, torch.inf))

    @pytest.mark.parametrize("use_mask", [False, True])
    def test_forward(self, use_mask: bool) -> None:
        self._test_forward(torch.device("cpu"), use_mask=use_mask)

    @pytest.mark.parametrize("use_mask", [False, True])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda not available.")
    def test_forward_cuda(self, use_mask: bool) -> None:
        self._test_forward(torch.device("cuda"), use_mask=use_mask)

    def test_images_to_tokens(self) -> None:
        torch.manual_seed(0)
        vit = self._vit()
        backbone = MAEBackbone.from_vit(vit)
        images = torch.rand(2, 3, 224, 224)
        assert torch.all(
            vit._process_input(images)
            == backbone.images_to_tokens(images, prepend_class_token=False)
        )


@pytest.mark.skipif(
    not dependency.torchvision_vit_available(),
    reason="Torchvision ViT not available",
)
class TestMAEDecoder:
    def test_init(self) -> None:
        MAEDecoder(
            seq_length=50,
            num_layers=2,
            num_heads=4,
            embed_input_dim=128,
            hidden_dim=256,
            mlp_dim=256 * 4,
            out_dim=3 * 32**2,
        )

    def _test_forward(
        self, device: torch.device, batch_size: int = 8, seed: int = 0
    ) -> None:
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
        assert list(predictions.shape) == expected_shape

        # output must have reasonable numbers
        assert torch.all(torch.not_equal(predictions, torch.inf))

    def test_forward(self) -> None:
        self._test_forward(torch.device("cpu"))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda not available.")
    def test_forward_cuda(self) -> None:
        self._test_forward(torch.device("cuda"))
