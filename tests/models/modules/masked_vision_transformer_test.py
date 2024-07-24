from abc import ABC, abstractmethod
from typing import Optional

import pytest
import torch
from pytest_mock import MockerFixture
from torch.nn import Parameter

from lightly.models import utils
from lightly.models.modules.masked_vision_transformer import MaskedVisionTransformer


class MaskedVisionTransformerTest(ABC):
    """Common tests for all classes implementing MaskedVisionTransformer."""

    @abstractmethod
    def get_masked_vit(
        self,
        patch_size: int,
        depth: int,
        num_heads: int,
        embed_dim: int = 768,
        class_token: bool = True,
        reg_tokens: int = 0,
        mask_token: Optional[Parameter] = None,
        antialias: bool = True,
        weight_initialization: str = "",
    ) -> MaskedVisionTransformer: ...

    @abstractmethod
    def test__init__mask_token(self, mask_token: Optional[Parameter]) -> None: ...

    @abstractmethod
    def test__init__weight_initialization(self, mocker: MockerFixture) -> None: ...

    @abstractmethod
    def test__init__weight_initialization__skip(
        self, mocker: MockerFixture
    ) -> None: ...

    def test_sequence_length(self) -> None:
        # TODO(Guarin, 07/2024): Support reg_tokens > 0 and test the sequence length
        # with reg_tokens > 0.
        model = self.get_masked_vit(
            patch_size=32,
            depth=1,
            num_heads=1,
            class_token=True,
            reg_tokens=0,
        )

        # 49 for img_height//patch_size * img_width//patch_size
        # 1 for the class token
        # 0 for the reg tokens
        expected_sequence_length = 49 + 1 + 0
        assert model.sequence_length == expected_sequence_length

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("mask_ratio", [None, 0.6])
    def test_forward(self, device: str, mask_ratio: Optional[float]) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available.")

        torch.manual_seed(0)
        batch_size = 8
        embed_dim = 768
        model = self.get_masked_vit(
            patch_size=32, depth=2, num_heads=2, embed_dim=embed_dim
        ).to(device)
        images = torch.rand(batch_size, 3, 224, 224).to(device)

        idx_keep = None
        if mask_ratio is not None:
            idx_keep, _ = utils.random_token_mask(
                size=(batch_size, model.sequence_length),
                device=device,
                mask_ratio=mask_ratio,
            )

        class_tokens = model(images=images, idx_keep=idx_keep)

        # output shape must be correct
        assert class_tokens.shape == (batch_size, embed_dim)
        # output must have reasonable numbers
        assert torch.all(torch.isfinite(class_tokens))

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "mask_ratio, expected_sequence_length", [(None, 50), (0.6, 20)]
    )
    def test_forward_intermediates(
        self, device: str, mask_ratio: Optional[float], expected_sequence_length: int
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available.")

        torch.manual_seed(0)
        batch_size = 8
        embed_dim = 768
        model = self.get_masked_vit(
            patch_size=32, depth=2, num_heads=2, embed_dim=embed_dim
        ).to(device)
        images = torch.rand(batch_size, 3, 224, 224).to(device)

        idx_keep = None
        if mask_ratio is not None:
            idx_keep, _ = utils.random_token_mask(
                size=(batch_size, model.sequence_length),
                device=device,
                mask_ratio=mask_ratio,
            )

        output, intermediates = model.forward_intermediates(
            images=images, idx_keep=idx_keep
        )
        expected_output = model.encode(images=images, idx_keep=idx_keep)

        # output shape must be correct
        assert output.shape == (batch_size, expected_sequence_length, embed_dim)
        # output should be same as from encode
        assert torch.allclose(output, expected_output)
        # intermediates must have reasonable numbers
        for intermediate in intermediates:
            assert intermediate.shape == (
                batch_size,
                expected_sequence_length,
                embed_dim,
            )
            assert torch.all(torch.isfinite(intermediate))

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "mask_ratio,expected_sequence_length", [(None, 50), (0.6, 20)]
    )
    def test_encode(
        self,
        device: str,
        mask_ratio: Optional[float],
        expected_sequence_length: int,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available.")

        torch.manual_seed(0)
        batch_size = 8
        embed_dim = 768
        model = self.get_masked_vit(
            patch_size=32, depth=2, num_heads=2, embed_dim=embed_dim
        ).to(device)
        images = torch.rand(batch_size, 3, 224, 224).to(device)

        idx_keep = None
        if mask_ratio is not None:
            idx_keep, _ = utils.random_token_mask(
                size=(batch_size, model.sequence_length),
                device=device,
                mask_ratio=mask_ratio,
            )

        tokens = model.encode(images=images, idx_keep=idx_keep)
        assert tokens.shape == (batch_size, expected_sequence_length, embed_dim)
        assert torch.all(torch.isfinite(tokens))

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_images_to_tokens(self, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available.")

        torch.manual_seed(0)
        model = self.get_masked_vit(
            patch_size=32, depth=2, num_heads=2, embed_dim=768
        ).to(device)
        images = torch.rand(2, 3, 224, 224).to(device)
        tokens = model.images_to_tokens(images)
        assert tokens.shape == (2, 49, 768)
        assert torch.all(torch.isfinite(tokens))

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "class_token,reg_tokens,expected_sequence_length",
        [
            (False, 0, 49),
            (True, 0, 50),
            # (False, 2, 51), TODO(Guarin, 07/2024): Support reg_tokens > 0
            # (True, 2, 52), TODO(Guarin, 07/2024): Support reg_tokens > 0
        ],
    )
    def test_add_prefix_tokens(
        self,
        device: str,
        class_token: bool,
        reg_tokens: int,
        expected_sequence_length: int,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available.")

        torch.manual_seed(0)
        model = self.get_masked_vit(
            patch_size=32,
            depth=1,
            num_heads=1,
            class_token=class_token,
            reg_tokens=reg_tokens,
        ).to(device)
        x = torch.rand(2, 49, 768).to(device)
        assert model.add_prefix_tokens(x).shape == (2, expected_sequence_length, 768)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "class_token,reg_tokens,expected_sequence_length",
        [
            (False, 0, 49),
            (True, 0, 50),
            # (False, 2, 51), TODO(Guarin, 07/2024): Support reg_tokens > 0
            # (True, 2, 52), TODO(Guarin, 07/2024): Support reg_tokens > 0
        ],
    )
    def test_add_pos_embed(
        self,
        device: str,
        class_token: bool,
        reg_tokens: int,
        expected_sequence_length: int,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available.")

        torch.manual_seed(0)
        model = self.get_masked_vit(
            patch_size=32,
            depth=1,
            num_heads=1,
            class_token=class_token,
            reg_tokens=reg_tokens,
        ).to(device)
        x = torch.rand(2, model.sequence_length, 768).to(device)
        assert model.add_pos_embed(x).shape == (2, expected_sequence_length, 768)
