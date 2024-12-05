# Code adapted from https://github.com/keyu-tian/SparK/blob/main/pretrain/encoder.py

import torch
import torch.nn as nn


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
    )  # ii: bi, hi, wi


def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(
        self.sparse_mask.mask, H=x.shape[2], W=x.shape[3], returning_active_ex=True
    )  # (BCHW) *= (B1HW), mask the output of conv
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
