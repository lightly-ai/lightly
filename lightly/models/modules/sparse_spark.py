# Copyright (c) 2023 Keyu Tian
# Copyright (c) ByteDance, Inc. and its affiliates.

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.common_types import _size_2_t

from lightly.models.utils import random_token_mask


def is_pow2n(x: int) -> bool:
    """Check if an integer is a power of 2.

    Args:
        x: Integer to check.

    Returns:
        True if x is a power of 2, False otherwise.
    """
    return x > 0 and (x & (x - 1) == 0)


def coalesce_to_size_2_t(t: tuple[int, ...]) -> _size_2_t:
    """Convert a 1-tuple or 2-tuple to standard (H, W) format.

    Args:
        t: Tuple of length 1 or 2 containing integers.

    Returns:
        A 2-tuple (h, w). If input is 1-tuple, both dimensions are equal.

    Raises:
        ValueError: If tuple length is not 1 or 2.
    """
    if len(t) == 2:
        return t
    elif len(t) == 1:
        return t[0], t[0]
    else:
        raise ValueError(f"Invalid tuple length: {len(t)}; expected 1 or 2.")


_cur_active: torch.Tensor | None = (
    None  # B1ff - Global active/mask tensor tracked during forward passes
)


def _get_active_ex_or_ii(
    H: int, W: int, returning_active_ex: bool = True
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Get active indices or expanded active mask from global _cur_active.

    Converts the global _cur_active mask (shape B, 1, f, f) to a given spatial resolution (H, W).
    Uses repeat_interleave to expand the mask to match the target spatial dimensions.

    Args:
        H: Target height dimension.
        W: Target width dimension.
        returning_active_ex: If True, return expanded binary mask (B, 1, H, W).
                            If False, return tuple of nonzero indices (bi, hi, wi).

    Returns:
        If returning_active_ex=True: Tensor of shape (B, 1, H, W) with binary active mask.
        If returning_active_ex=False: Tuple of 3 tensors (batch_idx, height_idx, width_idx) for active positions.

    Note:
        Optimization opportunity: Consider using gather() for better performance (see TODO).
    """
    assert _cur_active is not None, (
        "_cur_active must be set before calling this function"
    )
    h_repeat, w_repeat = H // _cur_active.shape[-2], W // _cur_active.shape[-1]
    assert h_repeat > 0, (
        f"Target height {H} must be >= mask height {_cur_active.shape[-2]}"
    )
    assert w_repeat > 0, (
        f"Target width {W} must be >= mask width {_cur_active.shape[-1]}"
    )
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(
        w_repeat, dim=3
    )
    return (
        active_ex
        if returning_active_ex
        else active_ex.squeeze(1).nonzero(as_tuple=True)
    )


def sp_conv_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for sparse convolution/pooling layers.

    Applies the parent class forward operation and masks the output using the global
    active mask to zero out inactive spatial positions.

    Args:
        self: ConvTranspose2d, MaxPool2d, or AvgPool2d instance.
        x: Input tensor of shape (B, C, H, W).

    Returns:
        Masked output tensor of same shape as input, with inactive spatial positions zeroed.
    """
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)
    return x


def sp_bn_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for sparse batch normalization layers.

    Applies batch norm only to active (unmasked) spatial positions, efficiently handling
    sparse feature maps by extracting active features, normalizing them, and reconstructing.
    Uses 1D batch norm on flattened active features rather than standard 2D batch norm.

    Args:
        self: BatchNorm1d or SyncBatchNorm instance.
        x: Input tensor of shape (B, C, H, W) in channels_first format.

    Returns:
        Output tensor of same shape as input, with batch norm applied only to active positions.
    """
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)

    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[ii]
    nc = super(type(self), self).forward(nc)

    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw


class SparseConv2d(nn.Conv2d):
    """Sparse 2D convolution layer that respects active/mask regions.

    Overrides forward pass to apply global active mask to output, zeroing inactive regions.
    Uses function override pattern for efficiency rather than standard subclassing override.
    """

    forward = sp_conv_forward


class SparseMaxPooling(nn.MaxPool2d):
    """Sparse max pooling layer that respects active/mask regions.

    Overrides forward pass to apply global active mask to output, zeroing inactive regions.
    Uses function override pattern for efficiency rather than standard subclassing override.
    """

    forward = sp_conv_forward


class SparseAvgPooling(nn.AvgPool2d):
    """Sparse average pooling layer that respects active/mask regions.

    Overrides forward pass to apply global active mask to output, zeroing inactive regions.
    Uses function override pattern for efficiency rather than standard subclassing override.
    """

    forward = sp_conv_forward


class SparseBatchNorm2d(nn.BatchNorm1d):
    """Sparse batch normalization for 2D feature maps.

    Overrides forward pass to apply batch norm only to active (unmasked) positions.
    Internally converts to 1D batch norm for efficient processing of sparse features.
    Uses function override pattern for efficiency rather than standard subclassing override.
    """

    forward = sp_bn_forward


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    """Sparse synchronized batch normalization for 2D feature maps.

    Overrides forward pass to apply synchronized batch norm only to active (unmasked) positions.
    Internally converts to 1D batch norm for efficient processing of sparse features.
    Uses function override pattern for efficiency rather than standard subclassing override.
    Recommended for distributed training scenarios.
    """

    forward = sp_bn_forward


class SparseConvNeXtLayerNorm(nn.LayerNorm):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.

    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    This sparse implementation only applies normalization to active (unmasked) spatial
    positions using indices from the global _cur_active mask.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim == 4:  # BHWC or BCHW
            if self.data_format == "channels_last":  # BHWC
                ii = _get_active_ex_or_ii(
                    H=input.shape[1], W=input.shape[2], returning_active_ex=False
                )
                nc = input[ii]
                nc = super().forward(nc)

                input = torch.zeros_like(input)
                input[ii] = nc
                return input
            else:  # channels_first, BCHW
                ii = _get_active_ex_or_ii(
                    H=input.shape[2], W=input.shape[3], returning_active_ex=False
                )
                bhwc = input.permute(0, 2, 3, 1)
                nc = bhwc[ii]
                nc = super().forward(nc)

                input = torch.zeros_like(bhwc)
                input[ii] = nc
                return input.permute(0, 3, 1, 2)
        else:  # BLC or BC
            raise NotImplementedError


class SparseConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block with sparse computation support.

    There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    We use (2) as we find it slightly faster in PyTorch. This sparse implementation always
    applies masking to the output.

    Args:
        dim: Number of input channels.
        drop_path: Stochastic depth rate. Default: 0.0
        layer_scale_init_value: Init value for Layer Scale. Default: 1e-6.
        ks: Kernel size for depthwise convolution. Default: 7.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        ks: int = 7,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=ks, padding=ks // 2, groups=dim)
        self.norm = SparseConvNeXtLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path: nn.Module = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)

        x = input + self.drop_path(x)
        return x


def dense_model_to_sparse(
    m: nn.Module, verbose: bool = False, sbn: bool = False
) -> nn.Module:
    """Recursively convert a dense model to sparse by replacing layer types.

    Handles Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d, SyncBatchNorm, and LayerNorm layers.
    Copies weight and state tensors to maintain the original model's parameters.

    Args:
        m: Original dense model or module.
        verbose: Whether to print conversion details. Default: False.
        sbn: Whether to use SyncBatchNorm instead of BatchNorm2d. Default: False.

    Returns:
        Sparse version of the input model with converted layers.

    Raises:
        NotImplementedError: If Conv1d layers are encountered.
    """
    oup = m
    if isinstance(m, nn.Conv2d):
        bias = m.bias is not None

        oup = SparseConv2d(
            m.in_channels,
            m.out_channels,
            kernel_size=coalesce_to_size_2_t(m.kernel_size),
            stride=coalesce_to_size_2_t(m.stride),
            padding=m.padding
            if isinstance(m.padding, str)
            else coalesce_to_size_2_t(m.padding),
            dilation=coalesce_to_size_2_t(m.dilation),
            groups=m.groups,
            bias=bias,
            padding_mode=m.padding_mode,
        )
        oup.weight.data.copy_(m.weight.data)

        if m.bias is not None and oup.bias is not None:
            oup.bias.data.copy_(m.bias.data)

    elif isinstance(m, nn.MaxPool2d):
        oup = SparseMaxPooling(
            m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            return_indices=m.return_indices,
            ceil_mode=m.ceil_mode,
        )
    elif isinstance(m, nn.AvgPool2d):
        oup = SparseAvgPooling(
            m.kernel_size,
            m.stride,
            m.padding,
            ceil_mode=m.ceil_mode,
            count_include_pad=m.count_include_pad,
            divisor_override=m.divisor_override,
        )
    elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        oup = (SparseSyncBatchNorm2d if sbn else SparseBatchNorm2d)(
            m.weight.shape[0],
            eps=m.eps,
            momentum=m.momentum,
            affine=m.affine,
            track_running_stats=m.track_running_stats,
        )
        oup.weight.data.copy_(m.weight.data)
        oup.bias.data.copy_(m.bias.data)
        oup.running_mean.data.copy_(m.running_mean.data)
        oup.running_var.data.copy_(m.running_var.data)
        oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
        if hasattr(m, "qconfig"):
            oup.qconfig = m.qconfig
    elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):
        oup = SparseConvNeXtLayerNorm(m.weight.shape[0], eps=m.eps)
        oup.weight.data.copy_(m.weight.data)
        oup.bias.data.copy_(m.bias.data)
    elif isinstance(m, (nn.Conv1d,)):
        raise NotImplementedError

    for name, child in m.named_children():
        oup.add_module(name, dense_model_to_sparse(child, verbose=verbose, sbn=sbn))
    del m
    return oup


class UNetBlock(nn.Module):
    """U-Net upsampling block with 2x spatial upsampling and conv refinement.

    Combines transposed convolution for 2x upsampling followed by residual convolutions
    with batch normalization and ReLU activation.

    Args:
        cin: Number of input channels.
        cout: Number of output channels.
        bn2d: Batch normalization layer class (e.g., nn.BatchNorm2d or nn.SyncBatchNorm).
    """

    def __init__(self, cin: int, cout: int, bn2d: type) -> None:
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(
            cin, cin, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False),
            bn2d(cin),
            nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
            bn2d(cout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply upsampling and convolution refinement.

        Args:
            x: Input feature tensor.

        Returns:
            Upsampled and refined feature tensor.
        """
        x = self.up_sample(x)
        return self.conv(x)


class LightDecoder(nn.Module):
    """Lightweight hierarchical decoder for feature map reconstruction.

    Applies a series of UNetBlocks to progressively upsample feature maps from deep to shallow,
    halving channels at each level according to a simple rule (width //= 2).
    Final projection outputs 3 channels for visualization/reconstruction.

    Args:
        up_sample_ratio: Total spatial upsampling ratio (must be power of 2).
        width: Base channel width at deepest level. Default: 768.
        sbn: Whether to use SyncBatchNorm (True) or BatchNorm2d (False). Default: True.
    """

    def __init__(
        self, up_sample_ratio: int, width: int = 768, sbn: bool = True
    ) -> None:
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2**i for i in range(n + 1)]
        bn2d = nn.SyncBatchNorm if sbn else nn.BatchNorm2d
        self.dec = nn.ModuleList(
            [
                UNetBlock(cin, cout, bn2d)
                for (cin, cout) in zip(channels[:-1], channels[1:])
            ]
        )
        self.proj = nn.Conv2d(channels[-1], 3, kernel_size=1, stride=1, bias=True)

        self.initialize()

    def forward(self, to_dec: list[torch.Tensor]) -> torch.Tensor:
        """Progressively upsample and combine feature maps.

        Args:
            to_dec: List of feature tensors from different scales (shallow to deep).

        Returns:
            Upsampled feature tensor with 3 output channels.
        """
        x = 0
        for i in range(len(self.dec)):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)

    def extra_repr(self) -> str:
        return f"width={self.width}"

    def initialize(self) -> None:
        """Initialize weights and biases using appropriate initialization schemes.

        Uses:
        - trunc_normal_ for Linear and Conv2d layers (std=0.02)
        - kaiming_normal_ for ConvTranspose2d layers
        - constant initialization for batch norm and layer norm (weight=1.0, bias=0.0)
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(
                m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)
            ):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


class SparKDensfiyBlock(nn.Module):
    """Block for densifying sparse features by filling masked regions with learned tokens.

    Applies normalization to sparse features, then uses learned mask tokens to fill inactive
    regions, finally projecting to target channel dimension.

    Args:
        e_width: Number of input channels (encoder width).
        d_width: Number of output channels (decoder width).
        densify_norm_str: Type of normalization ('bn', 'ln', or 'none'). Default: 'bn'.
        sbn: Whether to use SyncBatchNorm if densify_norm_str='bn'. Default: False.
        use_identity_proj: If True, use identity projection (assumes e_width == d_width). Default: False.
        kernel_size: Kernel size for projection convolution. Default: 3.
    """

    def __init__(
        self,
        e_width: int,
        d_width: int,
        densify_norm_str: str = "bn",
        sbn: bool = False,
        use_identity_proj: bool = False,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        # mask token
        p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
        trunc_normal_(p, mean=0, std=0.02, a=-0.02, b=0.02)
        self.mask_token = p

        # densify norm
        if densify_norm_str == "bn":
            self.densify_norm = (SparseSyncBatchNorm2d if sbn else SparseBatchNorm2d)(
                e_width
            )
        elif densify_norm_str == "ln":
            self.densify_norm = SparseConvNeXtLayerNorm(
                e_width, data_format="channels_first"
            )
        else:
            self.densify_norm = nn.Identity()

        # densify proj
        if use_identity_proj:
            self.densify_proj = nn.Identity()
            print(
                f"[SparKDensfiyBlock.__init__]: use nn.Identity() as densify_proj (e_width==d_width)"
            )
        else:
            densify_proj = nn.Conv2d(
                e_width,
                d_width,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=True,
            )
            print(
                f"[SparKDensfiyBlock.__init__]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}M)"
            )
            self.densify_proj = densify_proj

    def forward(
        self, bcff: torch.Tensor, cur_active: torch.Tensor
    ) -> torch.Tensor | None:
        """Densify sparse features by filling masked regions with learned tokens.

        Args:
            bcff: Sparse feature tensor of shape (B, C, H, W).
            cur_active: Boolean mask tensor of shape (B, 1, H, W) indicating active regions.

        Returns:
            Densified and projected feature tensor, or None if input is None.
        """
        if bcff is None:
            return None
        bcff = self.densify_norm(bcff)
        mask_tokens = self.mask_token.expand_as(bcff)
        bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)
        bcff = self.densify_proj(bcff)
        return bcff


class SparKDensifier(nn.Module):
    """Stack of densify blocks to convert sparse hierarchical features to dense features.

    Processes encoder feature maps from deepest to shallowest scale, applying SparKDensfiyBlock
    to each level. Handles the global _cur_active mask, dilating it at each upsampling level.

    Args:
        encoder_in_channels: List of channel numbers from encoder, shallow to deep.
        decoder_in_channel: Base channel number for decoder (at deepest scale).
        densify_norm_str: Type of normalization ('bn', 'ln', or 'none'). Default: 'bn'.
        sbn: Whether to use SyncBatchNorm if densify_norm_str='bn'. Default: False.
    """

    def __init__(
        self,
        encoder_in_channels: list[int],
        decoder_in_channel: int,
        densify_norm_str: str = "bn",
        sbn: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.encoder_in_channels = encoder_in_channels
        self.decoder_in_channel = decoder_in_channel
        d_width = decoder_in_channel
        for i, e_width in enumerate(encoder_in_channels[::-1]):
            use_identity = i == 0 and e_width == d_width
            kernel_size = 1 if i <= 0 else 3
            block = SparKDensfiyBlock(
                e_width=e_width,
                d_width=d_width,
                densify_norm_str=densify_norm_str,
                sbn=sbn,
                use_identity_proj=use_identity,
                kernel_size=kernel_size,
            )
            self.blocks.append(block)
            d_width //= 2

    def forward(self, fea_bcffs: list[torch.Tensor]) -> list[torch.Tensor | None]:
        """Convert sparse features to dense by filling masked regions.

        Args:
            fea_bcffs: List of feature tensors from encoder at different scales (shallow to deep).
                      Each tensor has shape (B, C, f, f) where f varies per scale.

        Returns:
            List of densified feature tensors for decoder processing (order reversed and dilated).
        """
        to_dec = []
        fea_bcffs = fea_bcffs[::-1]
        global _cur_active
        active_fmap_current = _cur_active
        for i, bcff in enumerate(fea_bcffs):
            if bcff is not None:
                bcff = self.blocks[i](bcff, active_fmap_current)
            to_dec.append(bcff)
            active_fmap_current = active_fmap_current.repeat_interleave(
                2, dim=2
            ).repeat_interleave(2, dim=3)
        return to_dec


class SparKMaskingOuptut(NamedTuple):
    """Output container for SparKMasker forward pass.

    Attributes:
        masked_bchw: Input image with masks applied at full resolution.
        per_level_mask: List of binary masks at each hierarchical level.
    """

    masked_bchw: torch.Tensor
    per_level_mask: list[torch.Tensor]


class SparKMasker(nn.Module):
    """Generates hierarchical random token masks for sparse feature processing.

    Creates a binary mask at patch level and expands it hierarchically to match
    feature maps at different spatial resolutions in the encoder.

    Args:
        feature_map_size: Tuple of (height, width) at patch/feature map level.
        downsample_ratio: Total downsampling ratio from input to feature map level.
        mask_ratio: Fraction of tokens to mask (make inactive). Default: 0.6.
    """

    def __init__(
        self,
        feature_map_size: tuple[int, int],
        downsample_ratio: int,
        mask_ratio: float = 0.6,
    ) -> None:
        super().__init__()
        self.fmap_h, self.fmap_w = feature_map_size
        self.downsample_ratio = downsample_ratio
        self.mask_ratio = mask_ratio

    def mask(self, B: int, device: torch.device) -> torch.Tensor:
        """Generate a random binary mask for features.

        Args:
            B: Batch size.
            device: Device to create the mask on.

        Returns:
            Boolean mask tensor of shape (B, 1, fmap_h, fmap_w) indicating active tokens.
        """
        h, w = self.fmap_h, self.fmap_w
        index_keep, _ = random_token_mask(
            size=(B, h * w), mask_ratio=self.mask_ratio, device=device
        )
        return (
            torch.zeros(B, 1, h * w, dtype=torch.bool, device=device)
            .scatter_(dim=2, index=index_keep.unsqueeze(1), value=True)
            .view(B, 1, h, w)
            .bool()
        )

    def forward(self, inp_bchw: torch.Tensor) -> SparKMaskingOuptut:
        """Generate hierarchical masks for the input.

        Creates masks at multiple scales from the patch level up to full input resolution.

        Args:
            inp_bchw: Input image tensor of shape (B, C, H, W).

        Returns:
            SparKMaskingOuptut containing:
            - masked_bchw: Input image masked at full resolution.
            - per_level_mask: List of masks at each hierarchical level.
        """
        global _cur_active
        _cur_active = self.mask(inp_bchw.shape[0], inp_bchw.device)
        active_b1ff = _cur_active.clone()
        downsample_ratio = self.downsample_ratio
        per_level_mask = [active_b1ff]
        for _ in range(int(math.log2(downsample_ratio))):
            previous_mask = per_level_mask[-1]
            active_b1cHcW = previous_mask.repeat_interleave(2, dim=2).repeat_interleave(
                2, dim=3
            )
            per_level_mask.append(active_b1cHcW)

        active_b1hw = per_level_mask[-1]
        masked_bchw = inp_bchw * active_b1hw
        return SparKMaskingOuptut(
            masked_bchw=masked_bchw, per_level_mask=per_level_mask
        )


class SparKOutputDecoder(nn.Module):
    """Decodes reconstructed patches back to image space and combines with original.

    Handles denormalization and unpatchifying of reconstructed patches, then performs
    per-pixel blending: uses original pixels where visible (active), reconstructed pixels
    where masked (inactive).

    Minimal configuration: only requires spatial properties (fmap_h, fmap_w, downsample_ratio).
    No encoder object needed.

    Args:
        fmap_h: Height of feature map at patch level.
        fmap_w: Width of feature map at patch level.
        downsample_ratio: Ratio of input image size to feature map size.
    """

    def __init__(self, fmap_h: int, fmap_w: int, downsample_ratio: int) -> None:
        super().__init__()
        self.fmap_h = fmap_h
        self.fmap_w = fmap_w
        self.downsample_ratio = downsample_ratio

    def unpatchify(self, bln: torch.Tensor) -> torch.Tensor:
        """Convert flattened patches back to spatial feature map format.

        Reverses the patchify operation: reshapes from (B, L*p*p, N) to (B, N, H, W)
        where p is the patch size (downsample_ratio).

        Args:
            bln: Flattened patches of shape (B, L, N).

        Returns:
            Spatial feature map of shape (B, C, H, W).
        """
        p = self.downsample_ratio
        h, w = self.fmap_h, self.fmap_w
        B, C = bln.shape[0], bln.shape[-1] // p**2
        bln = bln.reshape(shape=(B, h, w, p, p, C))
        bln = torch.einsum("bhwpqc->bchpwq", bln)
        bchw = bln.reshape(shape=(B, C, h * p, w * p))
        return bchw

    def forward(
        self,
        rec_patches: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
        inp_bchw: torch.Tensor,
        active_mask_full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct image by blending original and reconstructed regions.

        Denormalizes reconstructed patches, unpatchifies them, and performs pixel-wise
        blending based on the active mask: original where visible, reconstructed where masked.

        Args:
            rec_patches: Reconstructed patches of shape (B, L, N).
            mean: Per-patch mean of shape (B, L, 1).
            var: Per-patch standard deviation of shape (B, L, 1).
            inp_bchw: Original input image of shape (B, C, H, W).
            active_mask_full: Boolean mask of shape (B, 1, H, W) at full resolution.

        Returns:
            Tuple of:
            - inp_bchw: Original input image (unchanged).
            - masked_bchw: Input with inactive regions zeroed.
            - rec_or_inp: Blended result using original where visible, reconstructed where masked.
        """
        rec_bchw = self.unpatchify(rec_patches * var + mean)
        masked_bchw = inp_bchw * active_mask_full
        rec_or_inp = torch.where(active_mask_full, inp_bchw, rec_bchw)

        return inp_bchw, masked_bchw, rec_or_inp
