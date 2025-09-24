from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pytest
import torch
from pytest_mock import MockerFixture
from torch import Tensor
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
        pos_embed_initialization: str = "sincos",
    ) -> MaskedVisionTransformer:
        ...

    @abstractmethod
    def test__init__mask_token(self, mask_token: Optional[Parameter]) -> None:
        ...

    @abstractmethod
    def test__init__weight_initialization(self, mocker: MockerFixture) -> None:
        ...

    @abstractmethod
    def test__init__weight_initialization__skip(self, mocker: MockerFixture) -> None:
        ...

    def test__init__weight_initialization__invalid(self) -> None:
        with pytest.raises(ValueError):
            self.get_masked_vit(
                patch_size=32, depth=1, num_heads=1, weight_initialization="invalid"
            )

    def test__init__pos_embed_initialization(self, mocker: MockerFixture) -> None:
        mock_initialize_positional_embedding = mocker.spy(
            utils, "initialize_positional_embedding"
        )
        self.get_masked_vit(
            patch_size=32, depth=1, num_heads=1, pos_embed_initialization="learn"
        )
        mock_initialize_positional_embedding.assert_called_once()
        _, call_kwargs = mock_initialize_positional_embedding.call_args
        assert call_kwargs["strategy"] == "learn"

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

    @pytest.mark.slow
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "idx_mask_ratio, bool_mask_ratio",
        [(None, None), (0.6, None), (None, 0.6)],
    )
    @pytest.mark.parametrize("idx_keep_none", [False, True])
    def test_forward(
        self,
        device: str,
        idx_mask_ratio: Optional[float],
        bool_mask_ratio: Optional[float],
        idx_keep_none: bool,
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

        idx_keep, idx_mask, mask = self.get_masks(
            batch_size=batch_size,
            idx_mask_ratio=idx_mask_ratio,
            bool_mask_ratio=bool_mask_ratio,
            sequence_length=model.sequence_length,
            device=device,
            idx_keep_none=idx_keep_none,
        )

        class_tokens = model(
            images=images, idx_keep=idx_keep, idx_mask=idx_mask, mask=mask
        )

        # Output shape must be correct.
        assert class_tokens.shape == (batch_size, embed_dim)
        # Output must have reasonable numbers.
        assert torch.all(torch.isfinite(class_tokens))

    @pytest.mark.slow
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "idx_mask_ratio, bool_mask_ratio",
        [(None, None), (0.6, None), (None, 0.6)],
    )
    @pytest.mark.parametrize("idx_keep_none", [False, True])
    def test_forward_intermediates(
        self,
        device: str,
        idx_mask_ratio: Optional[float],
        bool_mask_ratio: Optional[float],
        idx_keep_none: bool,
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

        idx_keep, idx_mask, mask = self.get_masks(
            batch_size=batch_size,
            idx_mask_ratio=idx_mask_ratio,
            bool_mask_ratio=bool_mask_ratio,
            sequence_length=model.sequence_length,
            device=device,
            idx_keep_none=idx_keep_none,
        )

        output, intermediates = model.forward_intermediates(
            images=images, idx_keep=idx_keep, idx_mask=idx_mask, mask=mask
        )
        expected_output = model.encode(
            images=images, idx_keep=idx_keep, idx_mask=idx_mask, mask=mask
        )

        # Output shape must be correct.
        assert output.shape == expected_output.shape
        # Output should be same as from encode.
        assert torch.allclose(output, expected_output)
        # Intermediates must have reasonable numbers.
        for intermediate in intermediates:
            assert intermediate.shape == expected_output.shape
            assert torch.all(torch.isfinite(intermediate))

    @pytest.mark.slow
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "idx_mask_ratio, bool_mask_ratio",
        [(None, None), (0.6, None), (None, 0.6)],
    )
    @pytest.mark.parametrize("idx_keep_none", [False, True])
    def test_encode(
        self,
        device: str,
        idx_mask_ratio: Optional[float],
        bool_mask_ratio: Optional[float],
        idx_keep_none: bool,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available.")

        torch.manual_seed(0)
        batch_size = 8
        embed_dim = 768
        model = self.get_masked_vit(
            patch_size=32,
            depth=2,
            num_heads=2,
            embed_dim=embed_dim,
        ).to(device)
        images = torch.rand(batch_size, 3, 224, 224).to(device)

        idx_keep, idx_mask, mask = self.get_masks(
            batch_size=batch_size,
            idx_mask_ratio=idx_mask_ratio,
            bool_mask_ratio=bool_mask_ratio,
            sequence_length=model.sequence_length,
            device=device,
            idx_keep_none=idx_keep_none,
        )

        tokens = model.encode(
            images=images, idx_keep=idx_keep, idx_mask=idx_mask, mask=mask
        )
        # Output shape must be correct..
        assert tokens.ndim == 3
        assert tokens.shape[0] == batch_size
        # Sequence length depends on idx_keep.
        assert 20 <= tokens.shape[1] <= 50
        assert tokens.shape[2] == embed_dim
        #
        assert torch.all(torch.isfinite(tokens))

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        (
            "idx_mask_ratio, bool_mask_ratio, idx_keep_none, expected_sequence_length, "
            "expected_num_masked"
        ),
        [
            (None, None, False, 50, 0),
            (None, None, True, 50, 0),
            # No masked because idx_keep != None which only returns unmasked tokens.
            (0.6, None, False, 20, 0),
            (0.6, None, True, 50, 30),
            (None, 0.6, False, 50, 30),
            (None, 0.6, True, 50, 30),
        ],
    )
    def test_preprocess(
        self,
        device: str,
        idx_mask_ratio: Optional[float],
        bool_mask_ratio: Optional[float],
        idx_keep_none: bool,
        expected_sequence_length: int,
        expected_num_masked: int,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available.")

        torch.manual_seed(0)
        batch_size = 8
        embed_dim = 768
        # Create mask token with large value to check if it is used correctly.
        mask_token = Parameter(torch.zeros(1, 1, embed_dim) + 10_000)
        model = self.get_masked_vit(
            patch_size=32,
            depth=2,
            num_heads=2,
            embed_dim=embed_dim,
            mask_token=mask_token,
        ).to(device)
        images = torch.rand(batch_size, 3, 224, 224).to(device)

        idx_keep, idx_mask, mask = self.get_masks(
            batch_size=batch_size,
            idx_mask_ratio=idx_mask_ratio,
            bool_mask_ratio=bool_mask_ratio,
            sequence_length=model.sequence_length,
            device=device,
            idx_keep_none=idx_keep_none,
        )

        tokens = model.preprocess(
            images=images, idx_keep=idx_keep, idx_mask=idx_mask, mask=mask
        )
        assert tokens.shape == (batch_size, expected_sequence_length, embed_dim)
        # Check if the mask token was used correctly. Note that the actual value
        # can be smaller than 10_000 because the positional embedding is added. This is
        # why we check for 1_000 instead.
        assert (tokens > 1_000).sum() == batch_size * expected_num_masked * embed_dim

    def test_preprocess__fail_idx_mask_and_mask(self) -> None:
        batch_size = 8
        model = self.get_masked_vit(patch_size=32, depth=2, num_heads=2, embed_dim=768)
        _, idx_mask, mask = self.get_masks(
            batch_size=batch_size,
            idx_mask_ratio=0.6,
            bool_mask_ratio=0.6,
            sequence_length=model.sequence_length,
        )
        images = images = torch.rand(batch_size, 3, 224, 224)
        with pytest.raises(
            ValueError, match="idx_mask and mask cannot both be set at the same time"
        ):
            model.preprocess(images=images, idx_mask=idx_mask, mask=mask)

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
    def test_prepend_prefix_tokens(
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
        assert model.prepend_prefix_tokens(x).shape == (
            2,
            expected_sequence_length,
            768,
        )

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

    def get_masks(
        self,
        batch_size: int,
        idx_mask_ratio: Optional[float],
        bool_mask_ratio: Optional[float],
        sequence_length: int,
        device: Optional[str] = None,
        idx_keep_none: bool = False,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        idx_keep = None
        idx_mask = None
        if idx_mask_ratio is not None:
            idx_keep, idx_mask = utils.random_token_mask(
                size=(batch_size, sequence_length),
                mask_ratio=idx_mask_ratio,
                device=device,
            )
        if idx_keep_none:
            idx_keep = None

        mask = None
        if bool_mask_ratio is not None:
            # Create random boolean mask that has exactly bool_mask_ratio of values
            # set to true.
            n = int(batch_size * sequence_length)
            n_masked = int(n * bool_mask_ratio)
            mask = torch.randperm(n).reshape(batch_size, sequence_length)
            mask = mask < n_masked
            mask = mask.to(device).to(torch.bool)
            assert mask.sum() == n_masked
        return idx_keep, idx_mask, mask
