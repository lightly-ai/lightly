# Code adapted from https://github.com/keyu-tian/SparK/blob/main/pretrain/encoder.py and https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py

from typing import Literal, Optional

import torch
import torch.nn as nn


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
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  
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
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)

                    x = torch.zeros_like(x)
                    x[ii] = nc
                    return x
                else:
                    return super(SparseConvNeXtLayerNorm, self).forward(x)
            else: 
                if self.sparse:
                    ii = _get_active_ex_or_ii(
                        H=x.shape[2], W=x.shape[3], returning_active_ex=False
                    )
                    bhwc = x.permute(0, 2, 3, 1)
                    nc = bhwc[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)

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
                return super(SparseConvNeXtLayerNorm, self).forward(x)

    def __repr__(self):
        return (
            super(SparseConvNeXtLayerNorm, self).__repr__()[:-1]
            + f', ch={self.data_format.split("_")[-1]}, sp={self.sparse})'
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
        return super(SparseConvNeXtBlock, self).__repr__()[:-1] + f", sp={self.sparse})"
