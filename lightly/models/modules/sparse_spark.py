import math
import sys
from pprint import pformat
from typing import List, NamedTuple

import timm
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.common_types import _size_2_t

from lightly.models.utils import patchify, random_token_mask


def is_pow2n(x):
    return x > 0 and (x & (x - 1) == 0)


def coalesce_to_size_2_t(t: tuple[int, ...]) -> _size_2_t:
    if len(t) == 2:
        return t
    elif len(t) == 1:
        return t[0], t[0]
    else:
        raise ValueError(f"Invalid tuple length: {len(t)}; expected 1 or 2.")


_cur_active: torch.Tensor = None  # B1ff


# todo: try to use `gather` for speed?
def _get_active_ex_or_ii(H: int, W: int, returning_active_ex=True):
    h_repeat, w_repeat = H // _cur_active.shape[-2], W // _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(
        w_repeat, dim=3
    )
    return (
        active_ex
        if returning_active_ex
        else active_ex.squeeze(1).nonzero(as_tuple=True)
    )  # ii: bi, hi, wi


def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(
        H=x.shape[2], W=x.shape[3], returning_active_ex=True
    )  # (BCHW) *= (B1HW), mask the output of conv
    return x


def sp_bn_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)

    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[
        ii
    ]  # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(
        nc
    )  # use BN1d to normalize this flatten feature `nc`

    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward  # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    forward = sp_bn_forward  # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseConvNeXtLayerNorm(nn.LayerNorm):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self, normalized_shape, eps=1e-6, data_format="channels_last", sparse=True
    ):
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse

    def forward(self, input: torch.Tensor):
        if input.ndim == 4:  # BHWC or BCHW
            if self.data_format == "channels_last":  # BHWC
                if self.sparse:
                    ii = _get_active_ex_or_ii(
                        H=input.shape[1], W=input.shape[2], returning_active_ex=False
                    )
                    nc = input[ii]
                    nc = super().forward(nc)

                    input = torch.zeros_like(input)
                    input[ii] = nc
                    return input
                else:
                    return super(SparseConvNeXtLayerNorm, self).forward(input)
            else:  # channels_first, BCHW
                if self.sparse:
                    ii = _get_active_ex_or_ii(
                        H=input.shape[2], W=input.shape[3], returning_active_ex=False
                    )
                    bhwc = input.permute(0, 2, 3, 1)
                    nc = bhwc[ii]
                    nc = super().forward(nc)

                    input = torch.zeros_like(bhwc)
                    input[ii] = nc
                    return input.permute(0, 3, 1, 2)
                else:
                    u = input.mean(1, keepdim=True)
                    s = (input - u).pow(2).mean(1, keepdim=True)
                    input = (input - u) / torch.sqrt(s + self.eps)
                    input = (
                        self.weight[:, None, None] * input + self.bias[:, None, None]
                    )
                    return input
        else:  # BLC or BC
            if self.sparse:
                raise NotImplementedError
            else:
                return super().forward(input)

    def __repr__(self):
        return (
            super().__repr__()[:-1]
            + f", ch={self.data_format.split('_')[-1]}, sp={self.sparse})"
        )


class SparseConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self, dim, drop_path=0.0, layer_scale_init_value=1e-6, sparse=True, ks=7
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=ks, padding=ks // 2, groups=dim
        )  # depthwise conv
        self.norm = SparseConvNeXtLayerNorm(dim, eps=1e-6, sparse=sparse)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
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
        self.sparse = sparse

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(
            x
        )  # GELU(0) == (0), so there is no need to mask x (no need to `x *= _get_active_ex_or_ii`)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.sparse:
            x *= _get_active_ex_or_ii(
                H=x.shape[2], W=x.shape[3], returning_active_ex=True
            )

        x = input + self.drop_path(x)
        return x

    def __repr__(self):
        return super().__repr__()[:-1] + f", sp={self.sparse})"


def get_downsample_ratio_from_timm_model(model: nn.Module) -> int:
    return model.feature_info[-1]["reduction"]


def get_enc_feat_map_chs_from_timm_model(model: nn.Module) -> List[int]:
    return [fi["num_chs"] for fi in model.feature_info]


class SparseEncoder(nn.Module):
    """
    Converts a dense CNN model to a sparse CNN model by replacing standard layers

    Attributes:
        enc_feat_map_chs: List[int]: list of channel numbers of feature maps at different scales, in the order from shallow to deep


    """

    enc_feat_map_chs: List[int]

    def __init__(self, cnn, input_size, sbn=False, verbose=False):
        super().__init__()
        self.sp_cnn = SparseEncoder.dense_model_to_sparse(
            m=cnn, verbose=verbose, sbn=sbn
        )
        self.input_size, self.downsample_ratio, self.enc_feat_map_chs = (
            input_size,
            get_downsample_ratio_from_timm_model(cnn),
            get_enc_feat_map_chs_from_timm_model(cnn),
        )
        # feature-map spatial size (height, width) at encoder output (patch grid)
        self.fmap_h = input_size // self.downsample_ratio
        self.fmap_w = input_size // self.downsample_ratio

    @staticmethod
    def dense_model_to_sparse(m: nn.Module, verbose=False, sbn=False):
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
            oup.add_module(
                name,
                SparseEncoder.dense_model_to_sparse(child, verbose=verbose, sbn=sbn),
            )
        del m
        return oup

    def forward(self, x):
        return self.sp_cnn(x)


# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn2d):
        """
        a UNet block with 2x up sampling
        """
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

    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)


class LightDecoder(nn.Module):
    def __init__(
        self, up_sample_ratio, width=768, sbn=True
    ):  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [
            self.width // 2**i for i in range(n + 1)
        ]  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        bn2d = nn.SyncBatchNorm if sbn else nn.BatchNorm2d
        self.dec = nn.ModuleList(
            [
                UNetBlock(cin, cout, bn2d)
                for (cin, cout) in zip(channels[:-1], channels[1:])
            ]
        )
        self.proj = nn.Conv2d(channels[-1], 3, kernel_size=1, stride=1, bias=True)

        self.initialize()

    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)

    def extra_repr(self) -> str:
        return f"width={self.width}"

    def initialize(self) -> None:
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

        # Copyright (c) ByteDance, Inc. and its affiliates.


# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

pretrain_default_model_kwargs = {
    "your_convnet": dict(),
    "resnet50": dict(drop_path_rate=0.05),
    "resnet101": dict(drop_path_rate=0.08),
    "resnet152": dict(drop_path_rate=0.10),
    "resnet200": dict(drop_path_rate=0.15),
    "convnext_small": dict(sparse=True, drop_path_rate=0.2),
    "convnext_base": dict(sparse=True, drop_path_rate=0.3),
    "convnext_large": dict(sparse=True, drop_path_rate=0.4),
}


def build_sparse_encoder(
    name: str, input_size: int, sbn=False, drop_path_rate=0.0, verbose=False
):
    kwargs = pretrain_default_model_kwargs[name]
    if drop_path_rate != 0:
        kwargs["drop_path_rate"] = drop_path_rate
    print(f"[build_sparse_encoder] model kwargs={kwargs}")
    cnn = timm.create_model(name, **kwargs)

    return SparseEncoder(cnn, input_size=input_size, sbn=sbn, verbose=verbose)


class SparKDensfiyBlock(nn.Module):
    def __init__(
        self,
        e_width: int,
        d_width: int,
        densify_norm_str: str = "bn",
        sbn: bool = False,
        use_identity_proj: bool = False,
        kernel_size: int = 3,
    ):
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
                e_width, data_format="channels_first", sparse=True
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

    def forward(self, bcff: torch.Tensor, cur_active: torch.BoolTensor):
        if bcff is None:
            return None
        bcff = self.densify_norm(bcff)
        mask_tokens = self.mask_token.expand_as(bcff)
        bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)
        bcff = self.densify_proj(bcff)
        return bcff


class SparKDensifier(nn.Module):
    """
    A stack of SparKDensfiyBlocks to convert hierarchical sparse features to hierarchical dense features for decoding

    Args:
        encoder_in_channels: list of channel numbers of feature maps at different scales from the encoder, in the order from shallow to deep
        decoder_in_channel: channel number of the feature map at the deepest scale for decoding
        densify_norm_str: the type of normalization inside SparKDensfiyBlock; can be "bn", "ln" or "none"
        sbn: whether to use SyncBatchNorm (True) or BatchNorm (False) if densify_norm_str is "bn"
    """

    def __init__(
        self,
        encoder_in_channels: list[int],
        decoder_in_channel: int,
        densify_norm_str: str = "bn",
        sbn: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.encoder_in_channels = encoder_in_channels
        self.decoder_in_channel = decoder_in_channel
        d_width = decoder_in_channel
        for i, e_width in enumerate(encoder_in_channels[::-1]):
            # from the smallest feat map to the largest; i=0: the last feat map; i=1: the second last feat map ...
            # fork arguments that depend on the position (previously used idx inside the block)
            use_identity = i == 0 and e_width == d_width
            kernel_size = 1 if i <= 0 else 3
            # build densify block and append
            block = SparKDensfiyBlock(
                e_width=e_width,
                d_width=d_width,
                densify_norm_str=densify_norm_str,
                sbn=sbn,
                use_identity_proj=use_identity,
                kernel_size=kernel_size,
            )
            self.blocks.append(block)
            # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
            d_width //= 2

    def forward(self, fea_bcffs: List[torch.Tensor]):
        """
        Args:
            fea_bcffs: a list of feature maps at different scales from the encoder, in the order from shallow to deep; each feature map is a tensor of shape (B, C, f, f)
        """
        to_dec = []
        fea_bcffs = fea_bcffs[::-1]  # from the smallest feat map to the largest
        global _cur_active
        active_fmap_current = (
            _cur_active  # (B, 1, f, f), the mask map at the current scale
        )
        for i, bcff in enumerate(
            fea_bcffs
        ):  # from the smallest feature map to the largest
            if bcff is not None:
                bcff = self.blocks[i](bcff, active_fmap_current)
            to_dec.append(bcff)
            active_fmap_current = active_fmap_current.repeat_interleave(
                2, dim=2
            ).repeat_interleave(
                2, dim=3
            )  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)
        return to_dec


class SparKMaskingOuptut(NamedTuple):
    masked_bchw: torch.Tensor
    per_level_mask: List[torch.Tensor]


class SparKMasker(nn.Module):
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
        global _cur_active
        _cur_active = self.mask(inp_bchw.shape[0], inp_bchw.device)  # (B, 1, f, f)
        active_b1ff = _cur_active.clone()
        downsample_ratio = self.downsample_ratio
        per_level_mask = [active_b1ff]
        for i in range(int(math.log2(downsample_ratio))):
            previous_mask = per_level_mask[-1]
            active_b1cHcW = previous_mask.repeat_interleave(2, dim=2).repeat_interleave(
                2, dim=3
            )  # (B, 1, f*ds, f*ds)
            per_level_mask.append(active_b1cHcW)

        active_b1hw = per_level_mask[
            -1
        ]  # the mask map at the deepest scale (the smallest feature map), which would be used for masking the input to the encoder; shape: (B, 1, f, f)
        masked_bchw = inp_bchw * active_b1hw
        return SparKMaskingOuptut(
            masked_bchw=masked_bchw, per_level_mask=per_level_mask
        )


class SparK(nn.Module):
    def __init__(
        self,
        sparse_encoder: SparseEncoder,
        dense_decoder: LightDecoder,
        mask_ratio: float = 0.6,
        densify_norm: str = "bn",
        sbn=False,
    ):
        super().__init__()
        # spatial and size info moved to SparseEncoder
        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder
        self.masker = SparKMasker(
            feature_map_size=(self.sparse_encoder.fmap_h, self.sparse_encoder.fmap_w),
            downsample_ratio=self.sparse_encoder.downsample_ratio,
            mask_ratio=mask_ratio,
        )
        self.densifier = SparKDensifier(
            encoder_in_channels=self.sparse_encoder.enc_feat_map_chs,
            decoder_in_channel=self.dense_decoder.width,
            densify_norm_str=densify_norm.lower(),
            sbn=sbn,
        )
        print(
            f"[SparK.__init__] dims of mask_tokens={tuple(b.mask_token.numel() for b in self.densifier.blocks)}"
        )

    def forward(
        self,
        inp_bchw: torch.Tensor,
        vis=False,
    ):
        # step1. Mask
        mask_out: SparKMaskingOuptut = self.masker(inp_bchw)
        masked_bchw, per_level_mask = mask_out
        active_b1fHfW = per_level_mask[0]
        active_b1hw = per_level_mask[-1]
        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcffs: List[torch.Tensor] = self.sparse_encoder(masked_bchw)
        # step3. Densify: get hierarchical dense features for decoding
        to_dec = self.densifier(fea_bcffs)
        # step4. Decode and reconstruct
        rec_bchw = self.dense_decoder(to_dec)
        inp, rec = (
            self.patchify(inp_bchw),
            self.patchify(rec_bchw),
        )  # inp and rec: (B, L = f*f, N = C*downsample_raito**2)
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
        inp = (inp - mean) / var
        l2_loss = ((rec - inp) ** 2).mean(
            dim=2, keepdim=False
        )  # (B, L, C) ==mean==> (B, L)

        non_active = (
            active_b1fHfW.logical_not().int().view(active_b1fHfW.shape[0], -1)
        )  # (B, 1, f, f) => (B, L)
        recon_loss = l2_loss.mul_(non_active).sum() / (
            non_active.sum() + 1e-8
        )  # loss only on masked (non-active) patches

        if vis:
            masked_bchw = inp_bchw * active_b1hw
            rec_bchw = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_b1hw, inp_bchw, rec_bchw)
            return inp_bchw, masked_bchw, rec_or_inp
        else:
            return recon_loss

    def patchify(self, bchw):
        p = self.sparse_encoder.downsample_ratio
        return patchify(bchw, p)  # (B, L=f*f, N=C*p*p)

    def unpatchify(self, bln):
        p = self.sparse_encoder.downsample_ratio
        h, w = self.sparse_encoder.fmap_h, self.sparse_encoder.fmap_w
        B, C = bln.shape[0], bln.shape[-1] // p**2
        bln = bln.reshape(shape=(B, h, w, p, p, C))
        bln = torch.einsum("bhwpqc->bchpwq", bln)
        bchw = bln.reshape(shape=(B, C, h * p, w * p))
        return bchw

    def __repr__(self):
        return (
            f"\n"
            f"[SparK.config]: {pformat(self.get_config(), indent=2, width=250)}\n"
            f"[SparK.structure]: {super().__repr__().replace(SparK.__name__, '')}"
        )

    def get_config(self):
        return {
            # self
            "mask_ratio": self.mask_ratio,
            "densify_norm_str": self.densify_norm_str,
            "sbn": self.sbn,
            # enc
            "sparse_encoder.input_size": self.sparse_encoder.input_size,
            # dec
            "dense_decoder.width": self.dense_decoder.width,
        }

    def state_dict(
        self, destination=None, prefix="", keep_vars=False, with_config=False
    ) -> dict[str, torch.Tensor]:
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        if with_config:
            state["config"] = self.get_config()
        return state

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict=True
    ) -> dict[str, torch.Tensor]:
        config: dict = state_dict.pop("config", None)
        incompatible_keys = super().load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = f"[SparseMIM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={ckpt_v})"
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err, file=sys.stderr)
        return incompatible_keys
