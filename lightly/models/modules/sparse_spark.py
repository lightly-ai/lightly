# Code adapted from https://github.com/keyu-tian/SparK/blob/main/pretrain/encoder.py and https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import List, Optional, Tensor
from torchvision.models.convnext import CNBlock, LayerNorm2d

from torchvision.models import ResNet, ConvNeXt


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    Attributes:
        x:
            The input tensor.
        drop_prob:
            The drop probability of the path. Default: 0.
        training:
            Whether the model is in training mode. Default: False
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    Attributes:
        drop_prob:
            The drop probability of the path. Default: None.
    """

    def __init__(self, drop_prob: Optional[float] = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _get_active_ex_or_ii(
    mask: torch.BoolTensor, H: int, W: int, returning_active_ex: bool = True
):
    h_repeat, w_repeat = H // mask.shape[-2], W // mask.shape[-1]
    active_ex = mask.repeat_interleave(h_repeat, dim=2).repeat_interleave(
        w_repeat, dim=3
    )
    return (
        active_ex
        if returning_active_ex
        else active_ex.squeeze(1).nonzero(as_tuple=True)
    )


def sp_conv_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(
        self.sparse_mask.mask, H=x.shape[2], W=x.shape[3], returning_active_ex=True
    )
    return x


def sp_bn_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(
        self.sparse_mask.mask, H=x.shape[2], W=x.shape[3], returning_active_ex=False
    )

    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[ii]
    nc = super(type(self), self).forward(nc)

    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward


class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    forward = sp_bn_forward


class SparseConvNeXtLayerNorm(nn.LayerNorm):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).


    Attributes:
        normalized_shape:
            Input shape from an expected input of size
            normalized_shape or a single integer.
        eps:
            A value added to the denominator for numerical stability. Default: 1e-6.
        data_format:
            The ordering of the dimensions in the inputs. Default: "channels_last".
        sparse:
            Whether to use sparse computation. Default: True.

    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: Literal["channels_last", "channels_first"] = "channels_last",
        sparse: bool = True,
    ):
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse

    def forward(self, x):
        if x.ndim == 4:
            if self.data_format == "channels_last":
                if self.sparse:
                    ii = _get_active_ex_or_ii(
                        H=x.shape[1], W=x.shape[2], returning_active_ex=False
                    )
                    nc = x[ii]
                    nc = super().forward(nc)

                    x = torch.zeros_like(x)
                    x[ii] = nc
                    return x
                else:
                    return super().forward(x)
            else:
                if self.sparse:
                    ii = _get_active_ex_or_ii(
                        H=x.shape[2], W=x.shape[3], returning_active_ex=False
                    )
                    bhwc = x.permute(0, 2, 3, 1)
                    nc = bhwc[ii]
                    nc = super().forward(nc)

                    x = torch.zeros_like(bhwc)
                    x[ii] = nc
                    return x.permute(0, 3, 1, 2)
                else:
                    u = x.mean(1, keepdim=True)
                    s = (x - u).pow(2).mean(1, keepdim=True)
                    x = (x - u) / torch.sqrt(s + self.eps)
                    x = self.weight[:, None, None] * x + self.bias[:, None, None]
                    return x
        else:  # BLC or BC
            if self.sparse:
                raise NotImplementedError
            else:
                return super().forward(x)

    def __repr__(self):
        return (
            super().__repr__()[:-1]
            + f", ch={self.data_format.split('_')[-1]}, sp={self.sparse})"
        )


class SparseConvNeXtBlock(nn.Module):
    r"""
    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Attributes:
        dim:
            Number of input channels.
        drop_path:
            Stochastic depth rate. Default: 0.0
        layer_scale_init_value:
            Init value for Layer Scale. Default: 1e-6.
        sparse:
            Whether to use sparse computation. Default: True.
        kernel_size:
            Kernel size of depthwise convolution. Default: 7.
    """

    def __init__(
        self,
        dim,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        sparse: bool = True,
        kernel_size=7,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = SparseConvNeXtLayerNorm(dim, eps=1e-6, sparse=sparse)
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

# Code adapted from https://github.com/keyu-tian/SparK/blob/main/pretrain/models/resnet.py


class SparseMask:
    """
    Basic class to store the mask for the sparse model.
    """

    def __init__(self):
        self.mask: Union[Tensor, None] = None


class SparseEncoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        input_size: int,
        downsample_ratio: Optional[int] = None,
    ):
        """Sparse encoder as used by SparK [0]

        Default params are the ones explained in the original code base. The backbone is assumed
        to follow the same API as the ResNet or ConvNext models from torchvision.
        [0] Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling https://arxiv.org/abs/2301.03580

        Attributes:
            backbone:
                Backbone model to extract features from images. Should have both
                the methods get_downsample_ratio() and get_feature_map_channels()
                implemented.
            input_size:
                Size of the input image.
            downsample_ratio:

        """
        super().__init__()
        self.input_size = input_size
        if downsample_ratio is None:
            self.downsample_ratio = self.get_downsample_ratio(backbone)
        else:
            self.downsample_ratio = downsample_ratio

        self.enc_feat_map_chs = (
            self.get_feature_map_channels(backbone),
        )
        self.sparse_mask = SparseMask()
        self.sparse_backbone = self.dense_model_to_sparse(
            m=backbone, sparse_mask=self.sparse_mask
        )

    def forward(
        self, x: torch.Tensor, mask: torch.BoolTensor, hierarchical: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the sparse encoder.

        Args:
            x: The input tensor.
            mask: The mask to apply to the input tensor. This should be a binary mask the
                size of the smallest resolution feature map [0] of the backbone model.

        Returns:
            The output tensor.

        [0] SparK codebase https://github.com/keyu-tian/SparK/tree/main/pretrain#some-details-how-we-mask-images-and-how-to-set-the-patch-size
        """
        assert mask.shape == (
            1,
            1,
            self.input_size // self.downsample_ratio,
            self.input_size // self.downsample_ratio,
        ), "Mask shape must be (1, 1, H // downsample_ratio, W // downsample_ratio)"
        self.sparse_mask.mask = mask

        ls = []

        if isinstance(self.sparse_backbone, ConvNeXt):
            if hierarchical:
                for i in range(0,8,2):
                    x = self.sparse_backbone.features[i](x)
                    x = self.sparse_backbone.features[i+1](x)
                    ls.append(x)
            else:
                x = self.sparse_backbone.avgpool(x)
                x = self.sparse_backbone.classifier(x)
                return x

        elif isinstance(self.sparse_backbone, ResNet):
            x = self.sparse_backbone.conv1(x)
            x = self.sparse_backbone.bn1(x)
            x = self.sparse_backbone.act1(x)
            x = self.sparse_backbone.maxpool(x)

            if hierarchical:
                ls = []
                x = self.sparse_backbone.layer1(x)
                ls.append(x)
                x = self.sparse_backbone.layer2(x)
                ls.append(x)
                x = self.sparse_backbone.layer3(x)
                ls.append(x)
                x = self.sparse_backbone.layer4(x)
                ls.append(x)
                return ls
            else:
                x = self.sparse_backbone.global_pool(x)
                x = self.sparse_backbone.fc(x)
                return x
            
        else:
            raise NotImplementedError("Backbone not supported")

    def get_downsample_ratio(self, backbone: nn.Module) -> int:
        """
        Try to get the downsample ratio of the backbone model.

        Returns:
            The downsample ratio of the backbone model.
        """
        try:
            return backbone.get_downsample_ratio()
        except AttributeError:
            x = torch.randn(1, 3, self.input_size, self.input_size)
            out = nn.Sequential(*list(backbone.children())[:-2])(x)

            return self.input_size // out.shape[-1]


    def dense_model_to_sparse(
        self, m: nn.Module, sparse_mask: SparseMask, sync_batch_norm: bool = False
    ) -> nn.Module:
        """
        Convert a dense model to a sparse model.

        Args:
            m: The dense model to convert.
            sparse_mask: The sparse mask to use for the conversion.
            sync_batch_norm: Whether to convert BatchNorm2d to SyncBatchNorm.

        Returns:
            The sparse model.
        """
        with torch.no_grad():
            oup = m
            if isinstance(m, nn.Conv2d):
                m: nn.Conv2d
                bias = m.bias is not None
                oup = SparseConv2d(
                    m.in_channels,
                    m.out_channels,
                    kernel_size=m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    groups=m.groups,
                    bias=bias,
                    padding_mode=m.padding_mode,
                )
                oup.sparse_mask = sparse_mask
                oup.weight.copy_(m.weight)
                if bias:
                    oup.bias.copy_(m.bias)
            elif isinstance(m, nn.MaxPool2d):
                m: nn.MaxPool2d
                oup = SparseMaxPooling(
                    m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    return_indices=m.return_indices,
                    ceil_mode=m.ceil_mode,
                )
                oup.sparse_mask = sparse_mask
            elif isinstance(m, nn.AvgPool2d):
                m: nn.AvgPool2d
                oup = SparseAvgPooling(
                    m.kernel_size,
                    m.stride,
                    m.padding,
                    ceil_mode=m.ceil_mode,
                    count_include_pad=m.count_include_pad,
                    divisor_override=m.divisor_override,
                )
                oup.sparse_mask = sparse_mask
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m: nn.BatchNorm2d
                oup = (
                    SparseSyncBatchNorm2d
                    if isinstance(m, nn.SyncBatchNorm)
                    else SparseBatchNorm2d
                )(
                    m.weight.shape[0],
                    eps=m.eps,
                    momentum=m.momentum,
                    affine=m.affine,
                    track_running_stats=m.track_running_stats,
                )
                oup.sparse_mask = sparse_mask
                oup.weight.copy_(m.weight)
                oup.bias.copy_(m.bias)
                oup.running_mean.copy_(m.running_mean)
                oup.running_var.copy_(m.running_var)
                oup.num_batches_tracked.copy_(m.num_batches_tracked)
                if hasattr(m, "qconfig"):
                    oup.qconfig = m.qconfig
            elif isinstance(m, CNBlock):
                m: CNBlock
                oup = SparseConvNeXtBlock(
                    m.weight.shape[0],
                    m.layer_scale.
                )
            elif isinstance(m, (nn.Conv1d,)):
                raise NotImplementedError

            for name, child in m.named_children():
                oup.add_module(
                    name, self.dense_model_to_sparse(child, sparse_mask=sparse_mask)
                )
            del m
            return oup


import math
import sys
from pprint import pformat
from typing import List, Literal, Tuple, Union

import torch
import torch.nn as nn

from lightly.models.modules.spark import (
    SparseBatchNorm2d,
    SparseConvNeXtLayerNorm,
    SparseSyncBatchNorm2d,
)
from lightly.models.sparse.encoder import SparseResnet


def is_pow2n(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


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
    def __init__(self, up_sample_ratio, width=768, sync_batch_norm=True):
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2**i for i in range(n + 1)]
        bn2d = nn.SyncBatchNorm if sync_batch_norm else nn.BatchNorm2d
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

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
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


class SparK(nn.Module):
    """
    The SparK model as used by SparK [0]

    Default params are the ones explained in the original code base. The backbone is assumed
    to follow the same API as the ResNet models from torchvision or a ConvNext model.
    [0] Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling https://arxiv.org/abs/2301.03580

    Attributes:
        sparse_encoder:
            Sparse encoder to extract features from images. Should have both
            the methods get_downsample_ratio() and get_feature_map_channels()
            implemented.
        dense_decoder:
            Dense decoder to reconstruct the image from the sparse features.
        mask_ratio:
            Ratio of the image to mask. Default: 0.6
        densify_norm:
            Type of normalization to use for densification. Default: 'bn'
        sbn:
            Whether to use SyncBatchNorm. Default: False
    """

    def __init__(
        self,
        sparse_encoder: SparseResnet,
        dense_decoder: LightDecoder,
        mask_ratio: float = 0.6,
        densify_norm: Literal["batch_norm", "layer_norm", "identity"] = "bn",
        sbn: bool = False,
    ):
        super().__init__()
        input_size, downsample_ratio = (
            sparse_encoder.input_size,
            sparse_encoder.downsample_ratio,
        )
        self.downsample_ratio = downsample_ratio
        self.fmap_h, self.fmap_w = (
            input_size // downsample_ratio,
            input_size // downsample_ratio,
        )
        self.mask_ratio = mask_ratio
        self.len_keep = round(self.fmap_h * self.fmap_w * (1 - mask_ratio))

        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder

        self.sbn = sbn
        self.hierarchy = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str = densify_norm.lower()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()

        e_widths, d_width = (
            self.sparse_encoder.enc_feat_map_chs,
            self.dense_decoder.width,
        )
        e_widths: List[int]
        for i in range(self.hierarchy):
            e_width = e_widths.pop()
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            torch.nn.init.trunc_normal_(p, mean=0, std=0.02, a=-0.02, b=0.02)
            self.mask_tokens.append(p)

            if self.densify_norm_str == "batch_norm":
                densify_norm = (
                    SparseSyncBatchNorm2d if self.sbn else SparseBatchNorm2d
                )(e_width)
            elif self.densify_norm_str == "layer_norm":
                densify_norm = SparseConvNeXtLayerNorm(
                    e_width, data_format="channels_first", sparse=True
                )
            else:
                densify_norm = nn.Identity()
            self.densify_norms.append(densify_norm)

            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()
                print(
                    f"[SparK.__init__, densify {i + 1}/{self.hierarchy}]: use nn.Identity() as densify_proj"
                )
            else:
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv2d(
                    e_width,
                    d_width,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=True,
                )
                print(
                    f"[SparK.__init__, densify {i + 1}/{self.hierarchy}]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}M)"
                )
            self.densify_projs.append(densify_proj)
            d_width //= 2

        print(
            f"[SparK.__init__] dims of mask_tokens={tuple(p.numel() for p in self.mask_tokens)}"
        )

    def mask(self, B: int, device, generator=None):
        """
        Generate a mask for the input tensor

        Attributes:
            B:
                Batch size
            device:
                Device to put the mask on
            generator:
                Random number generator
        """
        h, w = self.fmap_h, self.fmap_w
        idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
        idx = idx[:, : self.len_keep].to(device)  # (B, len_keep)
        return (
            torch.zeros(B, h * w, dtype=torch.bool, device=device)
            .scatter_(dim=1, index=idx, value=True)
            .view(B, 1, h, w)
        )

    def forward(
        self,
        inp_bchw: torch.Tensor,
        active_b1ff: torch.BoolTensor = None,
        return_loss: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the SparK model

        Attributes:
            inp_bchw:
                Input tensor
            active_b1ff:
                Active mask
            return_loss:
                Whether to return the loss
        """
        if active_b1ff is None:
            active_b1ff: torch.BoolTensor = self.mask(
                inp_bchw.shape[0], inp_bchw.device
            )
        active_b1hw = active_b1ff.repeat_interleave(
            self.downsample_ratio, 2
        ).repeat_interleave(self.downsample_ratio, 3)
        masked_bchw = inp_bchw * active_b1hw

        fea_bcffs: List[torch.Tensor] = self.sparse_encoder.forward(
            masked_bchw, active_b1ff, hierarchical=True
        )
        fea_bcffs.reverse()

        cur_active = active_b1ff
        to_dec = []
        for i, bcff in enumerate(fea_bcffs):
            if bcff is not None:
                bcff = self.densify_norms[i](bcff)
                mask_tokens = self.mask_tokens[i].expand_as(bcff)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)
                bcff: torch.Tensor = self.densify_projs[i](bcff)
            to_dec.append(bcff)
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(
                2, dim=3
            )

        rec_bchw = self.dense_decoder(to_dec)
        inp, rec = self.patchify(inp_bchw), self.patchify(rec_bchw)
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
        inp = (inp - mean) / var
        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)

        non_active = active_b1ff.logical_not().int().view(active_b1ff.shape[0], -1)
        recon_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8)

        if return_loss:
            return recon_loss

        masked_bchw = inp_bchw * active_b1hw
        rec_bchw = self.unpatchify(rec * var + mean)
        rec_or_inp = torch.where(active_b1hw, inp_bchw, rec_bchw)
        return inp_bchw, masked_bchw, rec_or_inp

    def patchify(self, bchw):
        p = self.downsample_ratio
        h, w = self.fmap_h, self.fmap_w
        B, C = bchw.shape[:2]
        bchw = bchw.reshape(shape=(B, C, h, p, w, p))
        bchw = torch.einsum("bchpwq->bhwpqc", bchw)
        bln = bchw.reshape(shape=(B, h * w, C * p**2))
        return bln

    def unpatchify(self, bln):
        p = self.downsample_ratio
        h, w = self.fmap_h, self.fmap_w
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
            "mask_ratio": self.mask_ratio,
            "densify_norm_str": self.densify_norm_str,
            "sbn": self.sbn,
            "hierarchy": self.hierarchy,
            "sparse_encoder.input_size": self.sparse_encoder.input_size,
            "dense_decoder.width": self.dense_decoder.width,
        }

    def state_dict(
        self, destination=None, prefix="", keep_vars=False, with_config=False
    ):
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        if with_config:
            state["config"] = self.get_config()
        return state

    def load_state_dict(self, state_dict, strict=True):
        config: dict = state_dict.pop("config", None)
        incompatible_keys = super().load_state_dict(
            state_dict, strict=strict
        )
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